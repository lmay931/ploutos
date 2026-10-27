#!/usr/bin/env python3
"""
Splits the Ploutos SFT data into train/val/test by date.

Split by date rather than randomly so the model is trained on the past and
tested on the future. A random split would let it see later dates during
training and get tested on earlier ones, which is fine for learning the output
format but means the test results don't say anything about real predictive
ability.

    train : dates <  VAL_START
    val   : VAL_START <= dates < TEST_START
    test  : dates >= TEST_START   (held out for the final eval, don't tune on it)

It prints the distribution first so you can check train is big enough and
move the boundaries if the data is bunched up at one end.

Usage:
    python make_temporal_split.py data/sft_training_data.jsonl --outdir data/splits
    # check the report, change the boundaries if needed, then re-run with --write
    python make_temporal_split.py data/sft_training_data.jsonl --outdir data/splits --write
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PipelineConfig

_cfg = PipelineConfig()
# ---- boundaries come from PipelineConfig (can override with a YAML) ----
VAL_START  = _cfg.val_start    # "2024-01-01" by default
TEST_START = _cfg.test_start   # "2024-07-01" by default


def load_rows(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad JSON on line {ln}: {e}")
    return rows


def get_date(row):
    d = row.get("metadata", {}).get("date")
    if not d:
        raise SystemExit("A row is missing metadata.date - cannot split temporally.")
    return d  # ISO YYYY-MM-DD sorts lexicographically


def bucket(date_str):
    if date_str < VAL_START:
        return "train"
    if date_str < TEST_START:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--outdir", default="data/splits")
    ap.add_argument("--write", action="store_true",
                    help="actually write the split files (default: report only)")
    args = ap.parse_args()

    rows = load_rows(args.infile)
    print(f"Loaded {len(rows)} rows from {args.infile}\n")

    # ---- distribution report ----
    by_year = Counter()
    by_ym = Counter()
    for r in rows:
        d = get_date(r)
        by_year[d[:4]] += 1
        by_ym[d[:7]] += 1

    print("Rows per year:")
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]:>5}")
    print()

    # bucket counts under current boundaries
    buckets = defaultdict(list)
    for r in rows:
        buckets[bucket(get_date(r))].append(r)

    print(f"Boundaries: train < {VAL_START} <= val < {TEST_START} <= test")
    total = len(rows)
    for name in ("train", "val", "test"):
        n = len(buckets[name])
        pct = 100 * n / total if total else 0
        print(f"  {name:<5}: {n:>5}  ({pct:4.1f}%)")
    print()

    # ---- label balance etc. in each bucket ----
    print("Per-bucket label balance and experts_correct spread:")
    for name in ("train", "val", "test"):
        b = buckets[name]
        if not b:
            print(f"  {name}: EMPTY")
            continue
        gt = Counter(r["metadata"].get("ground_truth") for r in b)
        ec = Counter(r["metadata"].get("experts_correct") for r in b)
        tickers = len(set(r["metadata"].get("ticker") for r in b))
        dates = sorted(get_date(r) for r in b)
        print(f"  {name}: {len(b)} rows | {tickers} tickers | "
              f"{dates[0]}..{dates[-1]}")
        print(f"        ground_truth: {dict(gt)}")
        print(f"        experts_correct: {dict(sorted(ec.items()))}")
    print()

    # ---- warnings ----
    warn = []
    if len(buckets["train"]) < 800:
        warn.append(f"train has only {len(buckets['train'])} rows - light for SFT; "
                    f"consider moving VAL_START later to grow it.")
    if len(buckets["test"]) < 100:
        warn.append(f"test has only {len(buckets['test'])} rows - thin for a "
                    f"reliable out-of-sample accuracy/MCC estimate.")
    if len(buckets["val"]) < 50:
        warn.append(f"val has only {len(buckets['val'])} rows - checkpoint "
                    f"selection will be noisy.")
    # the same tickers appearing in every bucket is fine (same universe over
    # time), just flag it if one ticker makes up too much of a bucket
    for name in ("train", "test"):
        b = buckets[name]
        if b:
            tk = Counter(r["metadata"].get("ticker") for r in b)
            top_share = tk.most_common(1)[0][1] / len(b)
            if top_share > 0.10:
                warn.append(f"{name}: one ticker is {top_share:.0%} of the bucket "
                            f"- check for over-representation.")
    if warn:
        print("WARNINGS:")
        for w in warn:
            print(f"  ! {w}")
        print()
    else:
        print("No structural warnings.\n")

    if not args.write:
        print("Report only. Re-run with --write to create the split files "
              "(adjust VAL_START / TEST_START first if the above looks off).")
        return

    os.makedirs(args.outdir, exist_ok=True)
    for name in ("train", "val", "test"):
        out = os.path.join(args.outdir, f"{name}.jsonl")
        with open(out, "w") as f:
            for r in buckets[name]:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(buckets[name]):>5} rows -> {out}")
    print("\nDone. NOTE: test.jsonl is your sealed out-of-sample set. Do not use "
          "it for checkpoint selection, hyperparameter tuning, or any peeking "
          "before the final eval.")


if __name__ == "__main__":
    main()
