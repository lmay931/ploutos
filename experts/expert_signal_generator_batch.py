"""
Generates the expert signals using the Gemini batch API (google-genai SDK).

Usage:
  --mode create   build the requests, upload and start the batch job
                  (deterministic experts are computed and written straight away)
  --mode poll     poll the job until it succeeds/ fails/ is cancelled
  --mode merge    download the results and write the LLM signals into the master CSV

  --silo all|pm|sn|fn   which experts to run (price_macro/ sentiment/ fundamental
                        also work as aliases). default is all.

Price/macro is deterministic so it gets filled in during create and doesn't
need a batch request.

Sentiment follows Ploutos (Tong et al. 2024, sec 3.1.1): the LLM scores each
headline's 5-day price impact in [-1, +1] and the code takes the sign of the
mean (>= 0 -> BUY, < 0 -> SELL). I used to have hand-coded thresholds and
overrides here but they gave signals that contradicted their own rationale,
which is just label noise for the teacher. Weighing up context is left to the
aggregator (PloutosGPT). Rows with no scorable headlines get NEUTRAL, which
should never end up in PloutosGPT's output or Target_Label (those stay UP/DOWN).

Each expert gives its input and a direction, like Ploutos Table 11. The *_View
columns are template restatements of the signal (rule basis for price/macro,
aggregate score for sentiment, a pointer to Fund_Input for fundamental), so
they can't contradict the signal. The actual rationale is written later by
PloutosGPT with the ground truth available.
"""

import pandas as pd
import json
import os
import sys
import time
import argparse

from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.api_client import gemini_client
from config import PipelineConfig as _PipelineConfig
import experts  # noqa: F401  registers builtin experts into the registry
from experts.registry import all_experts
from experts.base import ResponseSchemaError

_cfg = _PipelineConfig()
GEMINI_BASIC_MODEL_NAME  = _cfg.expert_model       # overridden by --config at CLI entry
_HORIZON_DAYS            = _cfg.horizon_days        # forward return window
_NEWS_LOOKBACK_DAYS      = _cfg.news_lookback_days  # sentiment input window

BATCH_STATE_FILE = "data/batch_job_state.json"

ACTIVE_WAIT_ITERATIONS = 60
ACTIVE_WAIT_INTERVAL_SEC = 3

# ================================================================
# EXPERT DEFINITIONS (from the expert registry)
# ================================================================
# schemas, personas, input builders etc. live in the expert modules
# (experts/<id>.py), this just looks them up through the registry.
# SILO_DEFINITIONS wraps each Expert in the dict shape create/ merge use.
def _defn_for(expert):
    """Wrap an Expert in the dict that create/ merge use."""
    return {
        "signal_col": expert.signal_column,
        "view_col":   expert.view_column,
        "input_col":  expert.input_column,
        "agg_col":    getattr(expert, "agg_column", None),
        "mode":       "llm" if expert.is_llm else "deterministic",
        "builder":    expert.make_input,
        "compute":    (None if expert.is_llm else expert.compute),
        "persona":    expert.persona if expert.is_llm else None,
        "schema":     getattr(expert, "schema", None),
        "aggregates": getattr(expert, "aggregates", False),
        "expert":     expert,
    }


# all registered experts in registration order (pm, sn, fn, ...)
SILO_DEFINITIONS = {eid: _defn_for(e) for eid, e in all_experts().items()}

# --silo can be "all", an expert id, or one of the older names below
_LEGACY_SILO_ALIASES = {
    "price_macro": ["pm"],
    "sentiment":   ["sn"],
    "fundamental": ["fn"],
}
SILO_CLI_MAP = {"all": list(SILO_DEFINITIONS.keys())}
SILO_CLI_MAP.update({eid: [eid] for eid in SILO_DEFINITIONS})
SILO_CLI_MAP.update({k: v for k, v in _LEGACY_SILO_ALIASES.items()
                     if all(i in SILO_DEFINITIONS for i in v)})


def _ensure_signal_columns(df):
    for defn in SILO_DEFINITIONS.values():
        for col in (defn["signal_col"], defn["view_col"],
                    defn["input_col"], defn["agg_col"]):
            if col and col not in df.columns:
                df[col] = None


# ================================================================
# STEP 1: CREATE JOB (+ fill in deterministic experts)
# ================================================================
def create_batch_job(master_csv_path, silos, sample_size=None,
                     payload_path="data/batch_requests.jsonl"):
    print(f"[BATCH] Silo selection: {silos}")
    print("[BATCH] Loading master CSV...")
    master_df = pd.read_csv(master_csv_path)
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    _ensure_signal_columns(master_df)

    eligible = master_df[master_df['Target_Label'] != 'FLAT'].copy()

    # skip rows that already have a signal for every selected silo
    selected_signal_cols = [SILO_DEFINITIONS[s]["signal_col"] for s in silos]
    if all(c in eligible.columns for c in selected_signal_cols):
        already_done_mask = pd.Series(True, index=eligible.index)
        for c in selected_signal_cols:
            already_done_mask &= eligible[c].notna()
    else:
        already_done_mask = pd.Series(False, index=eligible.index)

    to_process = eligible[~already_done_mask].copy()
    skipped = int(already_done_mask.sum())
    if skipped:
        print(f"[BATCH] Skipping {skipped:,} rows that already have all "
              f"selected silo signals ({selected_signal_cols}).")

    if sample_size is not None:
        to_process = to_process.sample(
            n=min(sample_size, len(to_process)), random_state=42
        ).copy()
        print(f"[BATCH] Sample mode: {len(to_process)} rows.")
    else:
        print(f"[BATCH] Full mode: {len(to_process)} rows.")

    if len(to_process) == 0:
        print("[BATCH] Nothing to process.")
        return None

    det_silos = [s for s in silos if SILO_DEFINITIONS[s]["mode"] == "deterministic"]
    llm_silos = [s for s in silos if SILO_DEFINITIONS[s]["mode"] == "llm"]

    # deterministic silos, compute now and write straight into the CSV
    if det_silos:
        det_counts = {s: 0 for s in det_silos}
        for idx, row in to_process.iterrows():
            for s in det_silos:
                defn = SILO_DEFINITIONS[s]
                label, view = defn["compute"](row)
                master_df.loc[idx, defn["signal_col"]] = label
                master_df.loc[idx, defn["view_col"]] = view
                master_df.loc[idx, defn["input_col"]] = defn["builder"](row).strip()
                det_counts[s] += 1
        master_df.to_csv(master_csv_path, index=False)
        for s, n in det_counts.items():
            print(f"[BATCH] Deterministic silo '{s}' "
                  f"({SILO_DEFINITIONS[s]['signal_col']}): wrote {n:,} rows.")

    # no LLM silos selected so we're done
    if not llm_silos:
        print("[BATCH] No LLM silos selected; deterministic work complete. "
              "No batch job needed.")
        return None

    # LLM silos, build the batch requests
    requests = []
    inputs_by_id = {}

    model_name = (
        f"models/{GEMINI_BASIC_MODEL_NAME}"
        if not GEMINI_BASIC_MODEL_NAME.startswith("models/")
        else GEMINI_BASIC_MODEL_NAME
    )

    for _, row in to_process.iterrows():
        ticker = row['Ticker']
        date_str = row['Date'].strftime('%Y-%m-%d')
        base_id = f"{ticker}__{date_str}"

        for silo in llm_silos:
            defn = SILO_DEFINITIONS[silo]
            persona = defn["persona"]
            input_text = defn["builder"](row)
            req_id = f"{base_id}__{silo}"
            requests.append({
                "request_id": req_id,
                "request": {
                    "model": model_name,
                    "contents": [{"role": "user",
                                  "parts": [{"text": input_text.strip()}]}],
                    "system_instruction": {"parts": [{"text": persona.strip()}]},
                    "generation_config": {
                        "response_mime_type": "application/json",
                        # per-silo schema: sentiment returns scores,
                        # fundamental returns {signal}
                        "response_schema": defn["schema"],
                        # temp 0 for reproducibility (the paper uses 0 at eval
                        # too), matters most for sentiment since the sign
                        # comes straight from the scores
                        "temperature": 0.0
                    }
                }
            })
            inputs_by_id[req_id] = input_text.strip()

    print(f"[BATCH] Built {len(requests)} LLM requests "
          f"({len(to_process)} rows x {len(llm_silos)} LLM silo(s)).")

    os.makedirs(os.path.dirname(payload_path), exist_ok=True)
    with open(payload_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    inputs_path = payload_path.replace('.jsonl', '_inputs.json')
    with open(inputs_path, 'w') as f:
        json.dump(inputs_by_id, f)
    print(f"[BATCH] Snapshotted {len(inputs_by_id)} input texts -> {inputs_path}")

    print(f"[BATCH] Uploading {len(requests)} requests...")
    uploaded_file = gemini_client.files.upload(
        file=payload_path,
        config=types.UploadFileConfig(mime_type="application/jsonl")
    )

    print(f"[BATCH] Waiting for file to become ACTIVE "
          f"(up to {ACTIVE_WAIT_ITERATIONS * ACTIVE_WAIT_INTERVAL_SEC}s)...")
    for _ in range(ACTIVE_WAIT_ITERATIONS):
        state = gemini_client.files.get(name=uploaded_file.name).state.name
        if state == 'ACTIVE':
            break
        if state == 'FAILED':
            raise RuntimeError(f"File upload failed: {uploaded_file.name}")
        time.sleep(ACTIVE_WAIT_INTERVAL_SEC)
    else:
        raise RuntimeError(f"File ACTIVE timeout for {uploaded_file.name}.")

    batch_job = gemini_client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={"display_name": f"PloutosGen_Experts_{'-'.join(llm_silos)}_{int(time.time())}"}
    )

    print(f"\n[BATCH] Job created: {batch_job.name}")

    with open(BATCH_STATE_FILE, 'w') as f:
        json.dump({
            "job_name": batch_job.name,
            "master_csv_path": master_csv_path,
            "inputs_path": inputs_path,
            "silos": llm_silos,
        }, f)
    print(f"[BATCH] State saved to {BATCH_STATE_FILE}")

    return batch_job.name


# ================================================================
# STEP 2: POLL
# ================================================================
def poll_job(job_name, interval_seconds=60):
    while True:
        job = gemini_client.batches.get(name=job_name)
        status = job.state.name
        print(f"   [{time.strftime('%H:%M:%S')}] {status}")
        if status == 'JOB_STATE_SUCCEEDED':
            return True
        if status in ('JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            return False
        time.sleep(interval_seconds)


# ================================================================
# STEP 3: MERGE (LLM silos only, deterministic ones were written at create)
# ================================================================
def _strip_json_fences(text):
    """Gemini sometimes wraps the JSON in ```json ... ``` even with the json
    mime type set, so strip the fences before json.loads.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def merge_results(job_name, master_csv_path, inputs_path=None):
    job = gemini_client.batches.get(name=job_name)
    if job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"[MERGE] Job status: {job.state.name}. Run again later.")
        return

    output_filename = getattr(job.dest, 'file_name', None)
    if not output_filename:
        print(f"[ERROR] Could not extract file_name from job.dest: {job.dest}")
        return

    inputs_by_id = {}
    if inputs_path and os.path.exists(inputs_path):
        with open(inputs_path) as f:
            inputs_by_id = json.load(f)
        print(f"[MERGE] Loaded {len(inputs_by_id)} snapshotted inputs.")
    else:
        print(f"[MERGE] WARN: no snapshotted inputs at {inputs_path}. "
              f"Falling back to rebuilding from master_df.")

    print(f"[MERGE] Downloading file: {output_filename}...")
    content_bytes = gemini_client.files.download(file=output_filename)
    content_str = content_bytes.decode('utf-8')

    # req_id -> raw result dict (shape depends on the silo)
    parsed = {}
    failed_ids = []
    total_responses = 0

    for line in content_str.strip().split('\n'):
        if not line:
            continue
        total_responses += 1
        req_id = 'unknown'
        try:
            entry = json.loads(line)
            req_id = entry.get('request_id', 'unknown')
            candidate = entry['response']['candidates'][0]
            text = candidate['content']['parts'][0]['text']
            result = json.loads(_strip_json_fences(text))
            parsed[req_id] = result
        except Exception:
            failed_ids.append(req_id)

    print(f"[MERGE] Downloaded {total_responses} responses. "
          f"Parsed {len(parsed)} JSON payloads.")

    if not parsed:
        print("[MERGE] No parseable responses. Aborting before writing CSV.")
        if failed_ids:
            failed_path = "data/batch_failed_ids.json"
            os.makedirs(os.path.dirname(failed_path), exist_ok=True)
            with open(failed_path, 'w') as f:
                json.dump(failed_ids, f, indent=2)
            print(f"[MERGE] {len(failed_ids)} failed IDs -> {failed_path}")
        return

    master_df = pd.read_csv(master_csv_path)
    master_df['Date'] = pd.to_datetime(master_df['Date'])
    _ensure_signal_columns(master_df)

    # only LLM silos get written here, a deterministic silo turning up in the
    # results would be stale so it's skipped with a warning
    silo_defn_by_code = {
        code: defn for code, defn in SILO_DEFINITIONS.items()
        if defn["mode"] == "llm"
    }

    merged_count = 0
    orphan_count = 0
    schema_fail_ids = []
    per_silo_counts = {code: 0 for code in silo_defn_by_code}

    for req_id, result in parsed.items():
        try:
            rest, silo = req_id.rsplit('__', 1)
            ticker, d_str = rest.rsplit('__', 1)
        except ValueError:
            print(f"[MERGE] WARN: malformed request_id {req_id!r}, skipping.")
            continue

        if silo not in silo_defn_by_code:
            print(f"[MERGE] WARN: unexpected silo {silo!r} in {req_id} "
                  f"(deterministic or stale), skipping.")
            continue

        defn = silo_defn_by_code[silo]
        sig_c, view_c, inp_c = defn["signal_col"], defn["view_col"], defn["input_col"]
        builder = defn["builder"]
        expert = defn["expert"]

        # each expert parses its own response (sentiment aggregates the scores,
        # fundamental just returns the signal). schema mismatches get logged
        try:
            signal, view, agg = expert.parse_response(result)
        except ResponseSchemaError:
            schema_fail_ids.append(req_id)
            continue

        mask = (master_df['Ticker'] == ticker) & (master_df['Date'] == pd.to_datetime(d_str))
        if not mask.any():
            orphan_count += 1
            continue

        master_df.loc[mask, sig_c] = signal
        master_df.loc[mask, view_c] = view
        if agg is not None and defn["agg_col"]:
            master_df.loc[mask, defn["agg_col"]] = agg

        snapshot_text = inputs_by_id.get(req_id)
        if snapshot_text is not None:
            master_df.loc[mask, inp_c] = snapshot_text
        else:
            master_df.loc[mask, inp_c] = builder(master_df.loc[mask].iloc[0])

        merged_count += 1
        per_silo_counts[silo] += 1

    # save the failed ids (parse + schema failures)
    all_failed = failed_ids + schema_fail_ids
    if all_failed:
        failed_path = "data/batch_failed_ids.json"
        os.makedirs(os.path.dirname(failed_path), exist_ok=True)
        with open(failed_path, 'w') as f:
            json.dump({"json_parse_failures": failed_ids,
                       "schema_failures": schema_fail_ids}, f, indent=2)
        print(f"[MERGE] WARN: {len(failed_ids)} JSON-parse and "
              f"{len(schema_fail_ids)} schema failures -> {failed_path}")

    master_df.to_csv(master_csv_path, index=False)
    print(f"[MERGE] Wrote {merged_count} signals into {master_csv_path}.")
    for code, n in per_silo_counts.items():
        if n:
            print(f"   {code} ({SILO_DEFINITIONS[code]['signal_col']}): {n:,}")
    if orphan_count:
        print(f"[MERGE] WARN: {orphan_count} responses had no matching "
              f"(Ticker, Date) in master_df - these were discarded.")


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['create', 'poll', 'merge'], required=True)
    parser.add_argument('--job', type=str, default=None,
                        help="Override job_name (otherwise read from state file).")
    parser.add_argument('--csv', type=str, default="data/master_training_data.csv")
    parser.add_argument('--sample', type=int, default=None,
                        help="Process only N rows (sample mode).")
    parser.add_argument('--inputs', type=str, default=None,
                        help="Override inputs JSON path.")
    parser.add_argument(
        '--silo',
        choices=list(SILO_CLI_MAP.keys()),
        default='all',
        help="Which silo(s) to generate. 'price_macro' is deterministic and "
             "completes during --mode create.",
    )
    args = parser.parse_args()

    state = {}
    if os.path.exists(BATCH_STATE_FILE):
        with open(BATCH_STATE_FILE) as f:
            state = json.load(f)

    job_name = args.job or state.get('job_name')
    inputs_path = args.inputs or state.get('inputs_path')
    silos = SILO_CLI_MAP[args.silo]

    if args.mode == 'create':
        create_batch_job(args.csv, silos, args.sample)
    elif args.mode == 'poll':
        if not job_name:
            raise SystemExit("No --job provided and no state file found.")
        poll_job(job_name)
    elif args.mode == 'merge':
        if not job_name:
            raise SystemExit("No --job provided and no state file found.")
        merge_results(job_name, args.csv, inputs_path=inputs_path)