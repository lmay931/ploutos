"""
Batch API version of the Ploutos-style rearview-mirror teacher.

Output follows Ploutos Table 11: bullish rationales, bearish rationales and a
prediction paragraph under 100 words. There's no BUY/SELL field - the direction
is in the prediction text and only gets pulled out with extract_direction()
for diagnostics.

Compared to the old synchronous teacher_model.py:
  - uses the batch API (create/ poll/ merge), about half the cost and much faster
  - stratified sampling over (Year, Target_Label, experts_correct). random
    sampling gives too few 0/3 and 3/3 rows (~12-13% each) and DTW learns
    differently from all-wrong/ all-right/ split cases, so I want all 32 strata
    (4 years x 2 labels x 4 buckets) roughly balanced
  - cumulative: create skips (Ticker, Date) pairs already in the SFT file and
    merge adds new examples to it (deduped on ticker__date) rather than
    overwriting. payload/ snapshot paths are timestamped so two unmerged jobs
    don't overwrite each other.

The sentiment expert can return NEUTRAL when there are no usable headlines.
That counts as abstaining, so it's left out of experts_correct entirely and
the teacher is told not to write rationales for it. Target_Label and the final
prediction are still just UP/DOWN.

Only the price/macro *_View says anything useful - the sentiment/ fundamental
views are templated boilerplate - so build_data_context leaves them out (Table
11 has no view field anyway).

The teacher is only told the direction of the future move, not the size.
"""

import pandas as pd
import json
import os
import sys
import time
import argparse
from collections import defaultdict

from google.genai import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.api_client import gemini_client
from config import PipelineConfig as _PipelineConfig
from core.verbalizer import extract_direction
import experts  # noqa: F401  registers builtin experts into the registry
from experts.registry import all_experts, get_active

_cfg = _PipelineConfig()
GEMINI_ADVANCED_MODEL_NAME = _cfg.teacher_model   # overridden by --config at CLI entry
TEACHER_TEMPERATURE = _cfg.teacher_temperature

BATCH_STATE_FILE = "data/teacher_batch_job_state.json"
DEFAULT_OUTPUT_JSONL = _cfg.sft_jsonl

# expert columns and source tags come from the expert registry, so adding an expert
# doesn't need any change here. EXPERT_SIGNAL_COLS is only kept for other scripts that import it
EXPERT_SIGNAL_COLS = {eid: e.signal_column for eid, e in all_experts().items()}


def _source_clause(experts):
    """Builds the source expert list for the persona, e.g. 'A (descA), B (descB), or C (descC)'."""
    parts = [f"{e.source_tag} ({e.teacher_description})" for e in experts]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + ", or " + parts[-1]


# source tags + persona clause for the active experts, used by the schema and persona below
_ACTIVE_EXPERTS = get_active(_cfg)
_SOURCE_TAGS = [e.source_tag for e in _ACTIVE_EXPERTS]
_SOURCE_CLAUSE = _source_clause(_ACTIVE_EXPERTS)

ACTIVE_WAIT_ITERATIONS = 60
ACTIVE_WAIT_INTERVAL_SEC = 3

# NEUTRAL is left out on purpose - it's an abstention so it has no direction
SIGNAL_MAP = {'BUY': 'UP', 'SELL': 'DOWN'}

# 4 years x 2 labels x 4 experts_correct buckets
N_STRATA = 32

# if True, merge drops rows where the prediction text goes against the known
# direction. the teacher is told the answer so a contradiction is just a bad
# example and would train the student on the wrong thing. off by default
DROP_DIRECTION_MISMATCHES = False


# ================================================================
# TEACHER PERSONA  (system instruction)
# grounding rules live here, the per-row prompt only has the data and the
# rearview direction so the instructions aren't repeated
# ================================================================
#
# the output shape is set by STRICT_SCHEMA (constrained JSON decoding), so the
# persona has no markdown/ layout instructions - they'd clash with the schema.
# the "Bullish Rationales:\n- [SOURCE] ..." text is rebuilt from the JSON in
# merge_results() so the SFT format is the same either way
TEACHER_PERSONA = """
You are a seasoned stock market analyst. Your task is to extract the bullish
and bearish rationales for a company based on the expert signals and financials
provided, then provide an analysis and prediction for the company's stock
price movement for the upcoming period.

Each rationale is tagged with its source expert, one of: __SOURCE_CLAUSE__.

Rules:
1. Every rationale must be present in or inferable from the expert inputs
   provided. Do not invent external context, macro events, or news that are
   not in the inputs.
2. An expert may ABSTAIN (its prediction shows NEUTRAL). An abstaining expert
   has given no directional signal: do NOT write any rationale tagged to that
   expert, and do not treat its abstention as bullish or bearish. Build the
   rationales only from the experts that actually made a call.
3. The prediction must be the reasoned outcome of the analysis, not the
   starting point for it.
4. Rank rationales within each array by importance (most important first).
5. Where the experts disagreed, the prediction should briefly acknowledge
   which signal ultimately dominated and why.
6. Keep the prediction under 100 words. Its FIRST sentence must state the
   direction unambiguously using words like "up", "rise", "appreciation" or
   "down", "decline", "drop"; following sentences note which expert signals
   were most decisive. Downstream extraction reads only that first sentence.
""".replace("__SOURCE_CLAUSE__", _SOURCE_CLAUSE)

# ================================================================
# OUTPUT SCHEMA - source-tagged rationales + free-text prediction
# ================================================================
STRICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "bullish_rationales": {
            "type": "ARRAY",
            "description": "Bullish rationales grounded in the expert inputs, ranked by importance.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source": {"type": "STRING", "enum": _SOURCE_TAGS},
                    "rationale": {"type": "STRING", "description": "A single clinical bullish claim grounded in the data."}
                },
                "required": ["source", "rationale"]
            }
        },
        "bearish_rationales": {
            "type": "ARRAY",
            "description": "Bearish rationales grounded in the expert inputs, ranked by importance.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source": {"type": "STRING", "enum": _SOURCE_TAGS},
                    "rationale": {"type": "STRING", "description": "A single clinical bearish claim grounded in the data."}
                },
                "required": ["source", "rationale"]
            }
        },
        "prediction": {
            "type": "STRING",
            "description": (
                "Final prediction under 100 words. The first sentence must "
                "state the direction unambiguously using words like "
                "'rise'/'appreciation'/'upward' or 'decline'/'drop'/'downward'. "
                "Following sentences briefly note which expert signals were "
                "most decisive."
            )
        }
    },
    "required": ["bullish_rationales", "bearish_rationales", "prediction"]
}


# ================================================================
# HELPERS
# ================================================================
def _agrees(signal, truth):
    """True if a BUY/SELL signal matches the truth. NEUTRAL/ NaN never count as agreeing."""
    return SIGNAL_MAP.get(signal) == truth


def _is_directional(signal):
    """True if the expert actually called BUY/SELL rather than abstaining."""
    return signal in SIGNAL_MAP


def compute_experts_correct(df):
    """Adds experts_correct, experts_directional and Year columns to a copy of df.

    Abstaining (NEUTRAL) or missing experts aren't counted as right or wrong.
    experts_directional records how many experts actually made a call, so you
    can tell when a row's experts_correct is out of 2 rather than 3.
    """
    df = df.copy()
    sig_cols = [e.signal_column for e in get_active(_cfg)]

    def row_correct(r):
        return sum(_agrees(r[c], r['Target_Label']) for c in sig_cols)

    def row_directional(r):
        return sum(_is_directional(r[c]) for c in sig_cols)

    df['experts_correct'] = df.apply(row_correct, axis=1).astype(int)
    df['experts_directional'] = df.apply(row_directional, axis=1).astype(int)
    df['Year'] = df['Date'].dt.year
    return df


def load_done_keys(output_jsonl_path):
    """Returns the '{ticker}__{date}' keys already in the SFT file so create can skip them."""
    done = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    m = json.loads(line).get("metadata", {})
                    done.add(f"{m.get('ticker')}__{m.get('date')}")
                except Exception:
                    continue
    return done


def stratified_sample(eligible_df, sample_size, seed=33):
    """Samples evenly across (Year, Target_Label, experts_correct) strata.

    Thin strata just give what they have, so the total can come in a bit under
    sample_size. After a few cumulative runs more strata will run dry.
    """
    if len(eligible_df) == 0:
        print("[STRATIFY] No eligible rows remain (corpus may be saturated).")
        return eligible_df.copy()

    target_per_stratum = max(1, sample_size // N_STRATA)
    parts = []
    strata_report = []
    for (year, label, ec), group in eligible_df.groupby(['Year', 'Target_Label', 'experts_correct']):
        n_take = min(target_per_stratum, len(group))
        parts.append(group.sample(n=n_take, random_state=seed))
        strata_report.append((year, label, ec, len(group), n_take))

    sampled = pd.concat(parts).reset_index(drop=True)

    print(f"[STRATIFY] Target per stratum: {target_per_stratum} "
          f"(={sample_size}//{N_STRATA})")
    print(f"[STRATIFY] Strata filled: {len(strata_report)} / {N_STRATA}")
    thin = [s for s in strata_report if s[4] < target_per_stratum]
    if thin:
        print(f"[STRATIFY] {len(thin)} strata were thinner than target:")
        for year, label, ec, avail, took in sorted(thin):
            print(f"   {year} {label} ec={ec}: only {avail} available, took {took}")
    print(f"[STRATIFY] Total sampled: {len(sampled)} "
          f"(target was {sample_size})")
    return sampled


def build_data_context(row):
    """Builds the Table 11 style context: an Input block and a Prediction line per active expert.

    A 'Basis: <view>' line is only added for experts with include_basis set
    (just price/macro, the other views are boilerplate).
    """
    blocks = []
    for e in get_active(_cfg):
        inp = row.get(e.input_column, 'N/A')
        block = (f"[{e.prompt_title} Input]:\n{inp}\n\n"
                 f"[{e.prompt_title} Prediction]: {row[e.signal_column]}")
        if e.include_basis and e.view_column:
            block += f"\nBasis: {row.get(e.view_column, 'N/A')}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_rearview_prompt(row, data_context):
    """Per-row user prompt. Gives away the direction but not the magnitude."""
    ticker = row['Ticker']
    date_str = row['Date'].strftime('%Y-%m-%d')
    direction_word = "up" if row['Target_Label'] == "UP" else "down"
    return f"""{data_context}

[Analysis]:
Assume your prediction for {ticker} on {date_str} is {direction_word}.
Generate the bullish and bearish rationales from the expert information above
that justify this outcome, treating the prediction as the reasoned result of
the analysis rather than its premise. Use only experts that made a directional
call; ignore any expert whose prediction is NEUTRAL. Then conclude with a
prediction under 100 words whose first sentence states the direction
unambiguously."""


# ================================================================
# STEP 1: CREATE JOB
# ================================================================
def create_batch_job(master_data_path, sample_size,
                     payload_path=None,
                     output_jsonl_path=DEFAULT_OUTPUT_JSONL,
                     seed=33):
    print(f"[BATCH] Loading {master_data_path}...")
    df = pd.read_csv(master_data_path)
    df['Date'] = pd.to_datetime(df['Date'])

    active = get_active(_cfg)
    required = []
    for e in active:
        required += [e.signal_column, e.input_column]
        if e.include_basis and e.view_column:
            required.append(e.view_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Expert columns missing: {missing}. "
                         f"Run the expert signal generator first.")

    eligible = df[df['Target_Label'] != 'FLAT'].copy()

    # only drop rows where an expert that can't abstain has no signal. abstaining
    # experts (e.g. sentiment on a no-news day) are still a valid input
    must_have = [e.signal_column for e in active if not e.can_abstain]
    eligible = eligible.dropna(subset=must_have)
    # a missing cell would break the context so treat it as NEUTRAL
    for e in active:
        if e.can_abstain and e.signal_column in eligible.columns:
            eligible[e.signal_column] = eligible[e.signal_column].fillna('NEUTRAL')

    eligible = compute_experts_correct(eligible)

    print(f"[BATCH] {len(eligible):,} rows eligible (non-FLAT, required "
          f"non-abstain signals present).")
    for e in active:
        if e.can_abstain and e.signal_column in eligible.columns:
            n_abstain = int((eligible[e.signal_column] == 'NEUTRAL').sum())
            print(f"[BATCH] {e.name} expert abstains (NEUTRAL) on "
                  f"{n_abstain:,} of them.")

    # skip (Ticker, Date) pairs already in the SFT file
    done_keys = load_done_keys(output_jsonl_path)
    if done_keys:
        before = len(eligible)
        key_series = (eligible['Ticker'] + '__'
                      + eligible['Date'].dt.strftime('%Y-%m-%d'))
        eligible = eligible[~key_series.isin(done_keys)].copy()
        print(f"[BATCH] Excluded {before - len(eligible)} already-generated "
              f"examples ({len(done_keys)} in SFT file); "
              f"{len(eligible)} remain eligible.")

    if len(eligible) == 0:
        print("[BATCH] Nothing new to generate - corpus is saturated for the "
              "current master CSV. Add more eligible rows or check the SFT file.")
        return None

    sample_df = stratified_sample(eligible, sample_size, seed=seed)
    if len(sample_df) == 0:
        print("[BATCH] Stratified sample is empty. Aborting.")
        return None

    if payload_path is None:
        payload_path = f"data/teacher_batch_requests_{int(time.time())}.jsonl"

    requests = []
    inputs_by_id = {}
    meta_by_id = {}

    model_name = (
        f"models/{GEMINI_ADVANCED_MODEL_NAME}"
        if not GEMINI_ADVANCED_MODEL_NAME.startswith("models/")
        else GEMINI_ADVANCED_MODEL_NAME
    )

    for _, row in sample_df.iterrows():
        ticker = row['Ticker']
        date_str = row['Date'].strftime('%Y-%m-%d')
        req_id = f"{ticker}__{date_str}"

        data_context = build_data_context(row)
        prompt = build_rearview_prompt(row, data_context)

        requests.append({
            "request_id": req_id,
            "request": {
                "model": model_name,
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "system_instruction": {"parts": [{"text": TEACHER_PERSONA.strip()}]},
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": STRICT_SCHEMA,
                    "temperature": TEACHER_TEMPERATURE,
                }
            }
        })

        inputs_by_id[req_id] = data_context
        meta_by_id[req_id] = {
            "ticker": ticker,
            "date": date_str,
            "ground_truth": row['Target_Label'],
            "experts_correct": int(row['experts_correct']),
            "experts_directional": int(row['experts_directional']),
            "year": int(row['Year']),
        }

    os.makedirs(os.path.dirname(payload_path), exist_ok=True)
    with open(payload_path, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    inputs_path = payload_path.replace('.jsonl', '_inputs.json')
    with open(inputs_path, 'w') as f:
        json.dump({"inputs": inputs_by_id, "meta": meta_by_id}, f)
    print(f"[BATCH] Snapshotted {len(inputs_by_id)} contexts + metadata -> {inputs_path}")

    print(f"[BATCH] Uploading {len(requests)} requests...")
    uploaded_file = gemini_client.files.upload(
        file=payload_path,
        config=types.UploadFileConfig(mime_type="application/jsonl")
    )

    print(f"[BATCH] Waiting for file ACTIVE "
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
        config={"display_name": f"PloutosTeacher_{int(time.time())}"}
    )
    print(f"\n[BATCH] Job created: {batch_job.name}")

    with open(BATCH_STATE_FILE, 'w') as f:
        json.dump({
            "job_name": batch_job.name,
            "master_data_path": master_data_path,
            "inputs_path": inputs_path,
            "output_jsonl_path": output_jsonl_path,
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
# STEP 3: MERGE  (cumulative - folds into existing SFT JSONL)
# ================================================================
def merge_results(job_name, output_jsonl_path, inputs_path=None):
    job = gemini_client.batches.get(name=job_name)
    if job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"[MERGE] Job status: {job.state.name}. Run poll first.")
        return

    output_filename = getattr(job.dest, 'file_name', None)
    if not output_filename:
        print(f"[ERROR] Could not extract file_name from job.dest: {job.dest}")
        return

    if not (inputs_path and os.path.exists(inputs_path)):
        raise FileNotFoundError(
            f"Snapshot file not found at {inputs_path}. Cannot merge without "
            f"the input/metadata snapshot written at create time."
        )
    with open(inputs_path) as f:
        snapshot = json.load(f)
    inputs_by_id = snapshot["inputs"]
    meta_by_id = snapshot["meta"]
    print(f"[MERGE] Loaded {len(inputs_by_id)} snapshotted contexts.")

    # load the existing SFT examples first so new ones get added on top
    existing = {}
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                    m = ex.get("metadata", {})
                    key = f"{m.get('ticker')}__{m.get('date')}"
                    existing[key] = ex
                except Exception:
                    continue
        print(f"[MERGE] Loaded {len(existing)} existing SFT examples "
              f"for cumulative merge.")

    print(f"[MERGE] Downloading {output_filename}...")
    content_bytes = gemini_client.files.download(file=output_filename)
    content_str = content_bytes.decode('utf-8')

    added_or_refreshed = 0
    failed_ids = []
    dropped_mismatch = 0
    direction_match = 0
    direction_mismatch = 0
    direction_no_signal = 0
    ec_stats = defaultdict(lambda: {"total": 0, "match": 0, "mismatch": 0, "no_dir": 0})
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
            decision = json.loads(text)

            bullish = decision.get('bullish_rationales', [])
            bearish = decision.get('bearish_rationales', [])
            prediction_text = decision.get('prediction', '')

            if not prediction_text or 'bullish_rationales' not in decision:
                failed_ids.append(req_id)
                continue

            bullish_str = "\n".join(f"- [{r['source']}] {r['rationale']}" for r in bullish)
            bearish_str = "\n".join(f"- [{r['source']}] {r['rationale']}" for r in bearish)
            compiled_output = (
                f"Bullish Rationales:\n{bullish_str}\n\n"
                f"Bearish Rationales:\n{bearish_str}\n\n"
                f"Prediction: {prediction_text}"
            )

            meta = meta_by_id.get(req_id, {})
            ground_truth = meta.get('ground_truth')
            experts_correct = meta.get('experts_correct')

            # pull the direction out of the prose for diagnostics (and the optional gate)
            extracted = extract_direction(prediction_text, fallback_to_full=False)
            matches_truth = (extracted == ground_truth)

            if extracted is None:
                direction_no_signal += 1
                ec_stats[experts_correct]["no_dir"] += 1
            elif matches_truth:
                direction_match += 1
                ec_stats[experts_correct]["match"] += 1
            else:
                direction_mismatch += 1
                ec_stats[experts_correct]["mismatch"] += 1
            ec_stats[experts_correct]["total"] += 1

            # see DROP_DIRECTION_MISMATCHES. still counted in the diagnostics above
            if DROP_DIRECTION_MISMATCHES and extracted is not None and not matches_truth:
                dropped_mismatch += 1
                continue

            data_context = inputs_by_id.get(req_id, "")
            sft_example = {
                "messages": [
                    {"role": "system",
                     "content": "You are a seasoned quantitative AI portfolio manager."},
                    {"role": "user",
                     "content": (f"{data_context}\n\nBased on the expert inputs and "
                                 f"predictions above, generate the bullish and bearish "
                                 f"rationales and your final prediction.")},
                    {"role": "model", "content": compiled_output},
                ],
                "metadata": {
                    "ticker": meta.get('ticker'),
                    "date": meta.get('date'),
                    "ground_truth": ground_truth,
                    "teacher_model": GEMINI_ADVANCED_MODEL_NAME,
                    "experts_correct": experts_correct,
                    "experts_directional": meta.get('experts_directional'),
                    "year": meta.get('year'),
                    "prose_direction": extracted,
                    "prose_matches_truth": matches_truth,
                }
            }

            key = f"{meta.get('ticker')}__{meta.get('date')}"
            existing[key] = sft_example
            added_or_refreshed += 1

        except Exception:
            failed_ids.append(req_id)

    print(f"[MERGE] Downloaded {total_responses} responses. "
          f"Folded {added_or_refreshed} into the SFT set.")

    if failed_ids:
        failed_path = "data/teacher_batch_failed_ids.json"
        with open(failed_path, 'w') as f:
            json.dump(failed_ids, f, indent=2)
        print(f"[MERGE] WARN: {len(failed_ids)} responses failed parsing. "
              f"IDs -> {failed_path}")

    if not existing:
        print("[MERGE] No examples to write (nothing parsed, nothing pre-existing).")
        return

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    with open(output_jsonl_path, 'w') as f:
        for item in existing.values():
            f.write(json.dumps(item) + '\n')

    total_decided = direction_match + direction_mismatch + direction_no_signal
    def pct(x): return (x / total_decided * 100) if total_decided else 0.0

    print("-" * 60)
    print(f"TEACHER BATCH MERGE COMPLETE")
    print(f"   Folded this batch:      {added_or_refreshed}")
    if DROP_DIRECTION_MISMATCHES:
        print(f"   Dropped (mismatch gate):{dropped_mismatch}")
    print(f"   SFT file total now:     {len(existing)} examples")
    print(f"   Failed parse:           {len(failed_ids)}")
    print(f"   ---")
    print(f"   Direction diagnostics (this batch):")
    print(f"     prose matches truth:     {direction_match} ({pct(direction_match):.1f}%)")
    print(f"     prose contradicts truth: {direction_mismatch} ({pct(direction_mismatch):.1f}%)")
    print(f"     prose has no direction:  {direction_no_signal} ({pct(direction_no_signal):.1f}%)")
    print(f"   ---")
    print(f"   Match rate by experts_correct bucket (this batch):")
    for ec in sorted(k for k in ec_stats.keys() if k is not None):
        s = ec_stats[ec]
        if s["total"] == 0:
            continue
        m_rate = s["match"] / s["total"] * 100
        print(f"     ec={ec}: {s['total']} rows | match {s['match']} "
              f"({m_rate:.1f}%) | mismatch {s['mismatch']} | no-dir {s['no_dir']}")
    print(f"   Output:                 {output_jsonl_path}")
    print("-" * 60)


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['create', 'poll', 'merge'], required=True)
    parser.add_argument('--job', type=str, default=None)
    parser.add_argument('--csv', type=str, default="data/master_training_data.csv")
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument('--sample', type=int, default=5000,
                        help="Target sample size (stratified across 32 strata).")
    parser.add_argument('--seed', type=int, default=33,
                        help="Sampling seed. Vary across cumulative runs as "
                             "hygiene (exclusion already prevents duplicates).")
    parser.add_argument('--inputs', type=str, default=None)
    args = parser.parse_args()

    state = {}
    if os.path.exists(BATCH_STATE_FILE):
        with open(BATCH_STATE_FILE) as f:
            state = json.load(f)

    job_name = args.job or state.get('job_name')
    inputs_path = args.inputs or state.get('inputs_path')
    output_jsonl_path = state.get('output_jsonl_path') or args.out
    if args.mode == 'merge' and args.out != DEFAULT_OUTPUT_JSONL:
        output_jsonl_path = args.out

    if args.mode == 'create':
        create_batch_job(args.csv, args.sample,
                         output_jsonl_path=args.out, seed=args.seed)
    elif args.mode == 'poll':
        if not job_name:
            raise SystemExit("No --job and no state file found.")
        poll_job(job_name)
    elif args.mode == 'merge':
        if not job_name:
            raise SystemExit("No --job and no state file found.")
        merge_results(job_name, output_jsonl_path, inputs_path=inputs_path)