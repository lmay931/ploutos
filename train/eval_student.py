#!/usr/bin/env python3
"""
Out-of-sample evaluation of the Ploutos student on the test split.

Loads the base model + LoRA adapter and, for each test row, gives it just the
prompt (the three expert blocks) with no teacher rationale or ground truth, so
it has to write its own rationale and direction like it would when used for
real. Generation is greedy so runs are reproducible, and the direction is pulled
out of the text with extract_direction.

Reports accuracy and MCC overall and split by experts_correct (0-3), plus some
simple baselines (always UP, always DOWN, follow the expert majority) and how
often the student just agrees with the expert majority.

The experts_correct split is the main thing to look at. If the student only
does well on ec=2/3 (experts agreed and were right) it's just following the
experts - doing better than chance on ec=0/1, where the experts were mostly
wrong, is what would show it's actually learnt something.

MCC because accuracy on its own is misleading here - with ~55/45 labels always
predicting the majority class gets 55%. MCC is ~0 for constant or random
predictions.

Usage (from repo root, or slurm/eval.sbatch on the cluster):
    python train/eval_student.py --adapter out/sft_baseline/final --test data/splits/test.jsonl
    # baseline (untrained) reference:
    python train/eval_student.py --no-adapter --test data/splits/test.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.secrets  # noqa: F401  loads .env (HF_TOKEN) into os.environ
from config import PipelineConfig
from core.precision import describe_device, pick_dtype
from core.verbalizer import extract_direction
from core.io_utils import load_jsonl, to_messages, encode_chat
from core.metrics import report_block

cfg = PipelineConfig()
BASE_MODEL     = cfg.student_model
MAX_NEW_TOKENS = cfg.max_new_tokens


def encode_prompt(msgs_no_response, tokenizer):
    """Prompt to token ids, with the generation prompt added so the model
    starts the assistant turn."""
    return encode_chat(msgs_no_response, tokenizer, add_generation_prompt=True)


def expert_majority_direction(row):
    """Experts' majority vote, read from the '[... Expert Prediction]: BUY/SELL'
    lines in the user message. Used for the follow-the-experts baseline and the
    agreement check. None if there's a tie or no votes."""
    user = next(m["content"] for m in row["messages"] if m["role"] == "user")
    votes = []
    for line in user.splitlines():
        s = line.strip()
        if "Expert Prediction]:" in s:
            if s.endswith("BUY"):
                votes.append("UP")
            elif s.endswith("SELL"):
                votes.append("DOWN")
            # NEUTRAL/ anything else isn't counted
    if not votes:
        return None
    up = votes.count("UP")
    dn = votes.count("DOWN")
    if up == dn:
        return None  # tie
    return "UP" if up > dn else "DOWN"


def stratified_sample(rows, target_n, seed):
    """
    Sample ~target_n rows split evenly over (ground_truth x experts_correct), so
    every ec bucket has both UP and DOWN rows. Taking the first N rows (--limit)
    can easily be all DOWN since the file is in date order, which makes MCC and
    the bucket breakdown meaningless.
    """
    import random
    rng = random.Random(seed)
    cells = {}
    for r in rows:
        md = r["metadata"]
        key = (md.get("ground_truth"), md.get("experts_correct"))
        cells.setdefault(key, []).append(r)

    n_cells = len(cells)
    per_cell = max(1, target_n // n_cells)
    out = []
    for key, bucket in cells.items():
        rng.shuffle(bucket)
        out.extend(bucket[:per_cell])
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to trained LoRA adapter (omit / use --no-adapter "
                         "for the untrained base-model reference).")
    ap.add_argument("--no-adapter", action="store_true",
                    help="evaluate the base model with no fine-tuning.")
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the FIRST N rows (smoke test only - not "
                         "class-balanced; do not read metrics off this).")
    ap.add_argument("--stratified", type=int, default=None,
                    help="evaluate a class- and ec-bucket-balanced sample of "
                         "approximately this many rows. Use this for real "
                         "numbers on a budget instead of --limit.")
    ap.add_argument("--dump", default=None,
                    help="write per-row predictions to this JSONL (also used as "
                         "the resume/checkpoint file - see --resume).")
    ap.add_argument("--resume", action="store_true",
                    help="skip rows already present in the --dump file and append "
                         "the rest. Lets a timed-out eval continue where it left off.")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for stratified sampling reproducibility.")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    import random
    rows = load_jsonl(args.test)

    # ---- which rows to evaluate ----
    if args.stratified:
        rows = stratified_sample(rows, args.stratified, args.seed)
        print(f"Stratified sample: {len(rows)} rows "
              f"(balanced across UP/DOWN x experts_correct).", flush=True)
    elif args.limit:
        rows = rows[:args.limit]
        print("WARNING: --limit takes the FIRST N rows, which are temporally "
              "clustered and may be single-class. Use it only as a plumbing "
              "smoke test, NOT for metrics. Use --stratified for real numbers.")

    # ---- resume: skip rows already in the dump file ----
    done_keys = set()
    if args.resume and args.dump and os.path.exists(args.dump):
        with open(args.dump) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r.get("ticker"), r.get("date")))
                except json.JSONDecodeError:
                    continue
        before = len(rows)
        rows = [r for r in rows
                if (r["metadata"].get("ticker"), r["metadata"].get("date"))
                not in done_keys]
        print(f"Resume: {len(done_keys)} rows already done, "
              f"{len(rows)} of {before} remaining.", flush=True)

    print(f"Evaluating {len(rows)} test rows "
          f"({'BASE MODEL, no adapter' if args.no_adapter else args.adapter})\n",
          flush=True)

    dtype = pick_dtype()
    print(describe_device(dtype), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=dtype)

    if not args.no_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results = []        # dicts: pred, truth, ec, expert_vote, parsed_ok, ticker, date
    n_unparsed = 0
    dump_f = None
    if args.dump:
        dump_f = open(args.dump, "a" if (args.resume and os.path.exists(args.dump))
                      else "w")

    for i, row in enumerate(rows):
        msgs = to_messages(row)[:-1]  # drop the teacher's answer
        ids = torch.tensor([encode_prompt(msgs, tokenizer)], device=model.device)
        attn = torch.ones_like(ids)  # one row, no padding, so all ones
        with torch.no_grad():
            out = model.generate(ids, attention_mask=attn,
                                 max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

        # get the predicted direction from the generated text
        pred_block = gen.split("Prediction:", 1)[1] if "Prediction:" in gen else gen
        pred = extract_direction(pred_block)
        parsed_ok = pred is not None
        if not parsed_ok:
            n_unparsed += 1

        md = row["metadata"]
        rec = {
            "ticker": md.get("ticker"), "date": md.get("date"),
            "truth": md.get("ground_truth"),
            "ec": md.get("experts_correct"),
            "pred": pred,
            "expert_vote": expert_majority_direction(row),
            "parsed_ok": parsed_ok,
        }
        results.append(rec)
        if dump_f:
            dump_f.write(json.dumps({**rec, "generation": gen}) + "\n")
            dump_f.flush()  # write each row straight away so a timeout doesn't lose anything

        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(rows)} done", file=sys.stderr, flush=True)

    if dump_f:
        dump_f.close()

    # ---- if resuming, add the rows from earlier runs so the metrics cover
    #      the whole eval rather than just this run ----
    if args.resume and args.dump and done_keys:
        prior = []
        with open(args.dump) as f:
            seen = set()
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = (r.get("ticker"), r.get("date"))
                if k in seen:
                    continue  # skip duplicates
                seen.add(k)
                prior.append({
                    "ticker": r.get("ticker"), "date": r.get("date"),
                    "truth": r.get("truth"), "ec": r.get("ec"),
                    "pred": r.get("pred"), "expert_vote": r.get("expert_vote"),
                    "parsed_ok": r.get("parsed_ok",
                                       r.get("pred") is not None),
                })
        # prior already has this run's rows too since they were flushed to the
        # dump file, so just use it as the full set
        results = prior
        n_unparsed = sum(1 for r in results if not r["parsed_ok"])
        print(f"(resume) metrics computed over full set of {len(results)} rows.")

    # ---- metrics only use rows where a direction could be parsed ----
    parsed = [r for r in results if r["parsed_ok"] and r["truth"] in ("UP", "DOWN")]
    print(f"\nParsed a direction on {len(parsed)}/{len(results)} rows "
          f"({n_unparsed} unparseable - counted as abstentions, excluded from acc).")
    if n_unparsed:
        print("  (high unparse count => model isn't reliably emitting a verdict; "
              "inspect --dump output.)")

    student_pairs = [(r["pred"], r["truth"]) for r in parsed]

    print("\n================ STUDENT ================")
    report_block("OVERALL", student_pairs)

    print("\n  by experts_correct bucket (THE KEY VIEW):")
    for ec in (0, 1, 2, 3):
        bucket = [(r["pred"], r["truth"]) for r in parsed if r["ec"] == ec]
        report_block(f"ec={ec}", bucket)
    print("  ^ real edge = beating chance on ec=0/ec=1 (experts mostly WRONG).")
    print("    strong ec=2/ec=3 only = riding expert consensus.")

    # ---- simple baselines on the same parsed rows ----
    print("\n================ BASELINES (same rows) ================")
    truths = [r["truth"] for r in parsed]
    report_block("always-UP", [("UP", t) for t in truths])
    report_block("always-DOWN", [("DOWN", t) for t in truths])
    follow = [(r["expert_vote"], r["truth"]) for r in parsed
              if r["expert_vote"] in ("UP", "DOWN")]
    report_block("follow-expert-majority", follow)

    # ---- how often does the student just agree with the expert majority? ----
    both = [r for r in parsed if r["expert_vote"] in ("UP", "DOWN")]
    if both:
        agree = sum(1 for r in both if r["pred"] == r["expert_vote"]) / len(both)
        print(f"\nStudent agrees with expert-majority on {agree:.1%} of rows.")
        print("  ~100% => student is essentially a paraphrase of the experts.")
        print("  lower => student is making independent calls (good IF they're "
              "also more accurate than follow-expert-majority above).")

    print("\nReminder: test set is your sealed out-of-sample period. These are "
          "the numbers that matter for the 'is there edge' question. A student "
          "that doesn't beat the best dumb baseline here has no demonstrated edge.")


if __name__ == "__main__":
    main()
