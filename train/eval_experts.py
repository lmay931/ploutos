#!/usr/bin/env python3
"""
Scores each expert on its own against the test set (like Ploutos Tables 6-8).

No student or teacher here, just expert.signal(row) vs Target_Label for each
active expert. Useful for checking whether an expert gets above 50% before
spending anything on distilling it.

Rule-based experts (e.g. price/ macro) recompute signal() from the features, so
a new alpha-factor expert can be tested here without any LLM or GPU cost. The
LLM experts (sentiment/ fundamental) read their cached signal column, so
expert_signal_generator_batch.py needs to have been run first.

Reports accuracy and MCC overall and by experts_correct bucket. Look at both,
the paper found experts with decent accuracy but low MCC from always calling
the same direction.

The JSONL splits only have the teacher prompt + metadata, not the raw feature
columns, so the test split's (ticker, date) keys get joined back onto the
master CSV, which has the features, cached expert signals and Target_Label.

--ablation also gives the majority-vote accuracy/ MCC with each expert left out
in turn, to see how much each one adds to the consensus without retraining the
student. (Retraining a student without expert X is just a change to
config.active_experts and regenerating the data.)

Usage:
    python train/eval_experts.py
    python train/eval_experts.py --experts pm,fn          # ablate sentiment out
    python train/eval_experts.py --ablation               # + leave-one-out consensus
    python train/eval_experts.py --test data/splits/test.jsonl
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig                       # noqa: E402
from core.io_utils import load_jsonl                    # noqa: E402
from core.metrics import report_block                   # noqa: E402
import experts                                          # noqa: E402,F401  (registers)
from experts.registry import get, get_active            # noqa: E402

DIRECTIONS = ("UP", "DOWN")


def _test_keys(test_path):
    """(ticker, date) keys in a JSONL split, from the row metadata."""
    keys = set()
    for r in load_jsonl(test_path):
        md = r.get("metadata", {})
        if md.get("ticker") and md.get("date"):
            keys.add((md["ticker"], md["date"]))
    return keys


def _load_test_rows(master_csv, test_path):
    """Rows of the master CSV that are in the test split and have an UP/DOWN
    Target_Label."""
    df = pd.read_csv(master_csv)
    df["Date"] = pd.to_datetime(df["Date"])
    df["_key"] = df["Ticker"] + "__" + df["Date"].dt.strftime("%Y-%m-%d")
    keys = {f"{t}__{d}" for (t, d) in _test_keys(test_path)}
    df = df[df["_key"].isin(keys) & df["Target_Label"].isin(DIRECTIONS)].copy()
    return df


def _experts_correct(rows, active):
    """For each row, how many active experts called the right direction
    (NEUTRAL/ None count as wrong). Same order as rows.iterrows()."""
    out = []
    for _, row in rows.iterrows():
        truth = row["Target_Label"]
        out.append(sum(1 for e in active
                       if e.signal(row) in DIRECTIONS and e.signal(row) == truth))
    return out


def _consensus(row, active):
    """Majority vote of the active experts, None if tied or no votes."""
    votes = [e.signal(row) for e in active]
    votes = [v for v in votes if v in DIRECTIONS]
    if not votes:
        return None
    up, dn = votes.count("UP"), votes.count("DOWN")
    if up == dn:
        return None
    return "UP" if up > dn else "DOWN"


def per_expert_eval(rows, active):
    n_active = len(active)
    ec = _experts_correct(rows, active)
    row_list = [row for _, row in rows.iterrows()]

    print(f"\n================ PER-EXPERT STANDALONE (n={len(row_list)} test "
          f"rows, experts={[e.id for e in active]}) ================")
    for e in active:
        pairs, n_directional = [], 0
        for row in row_list:
            sig = e.signal(row)
            truth = row["Target_Label"]
            if sig in DIRECTIONS:
                n_directional += 1
                pairs.append((sig, truth))
        cov = n_directional / len(row_list) if row_list else 0.0
        print(f"\n[{e.id}] {e.name}  (coverage {cov:.1%}: made a directional "
              f"call on {n_directional}/{len(row_list)} rows)")
        report_block(f"{e.id} OVERALL", pairs)
        for b in range(n_active + 1):
            bucket = [(e.signal(r), r["Target_Label"])
                      for r, c in zip(row_list, ec)
                      if c == b and e.signal(r) in DIRECTIONS]
            report_block(f"{e.id} ec={b}", bucket)
    print("\n^ real edge = beating 50% / mcc>0 standalone. The paper's "
          "expectation is each current expert ~49% - near-chance.")


def consensus_eval(rows, active, ablation):
    row_list = [row for _, row in rows.iterrows()]

    def consensus_pairs(subset):
        out = []
        for row in row_list:
            c = _consensus(row, subset)
            if c in DIRECTIONS:
                out.append((c, row["Target_Label"]))
        return out

    print(f"\n================ CONSENSUS (majority vote) ================")
    report_block("active-set " + "+".join(e.id for e in active),
                 consensus_pairs(active))

    if ablation and len(active) > 1:
        print("\n  leave-one-out (marginal contribution of each expert):")
        for drop in active:
            subset = [e for e in active if e.id != drop.id]
            report_block(f"  drop {drop.id} -> [{'+'.join(e.id for e in subset)}]",
                         consensus_pairs(subset))
        print("  ^ compare each leave-one-out row to the active-set row above: a "
              "big drop in mcc when an expert is removed = that expert was "
              "carrying the consensus.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/splits/test.jsonl")
    ap.add_argument("--master", default="data/master_training_data.csv")
    ap.add_argument("--experts", default=None,
                    help="comma-separated expert ids to evaluate (overrides "
                         "config.active_experts), e.g. 'pm,fn'.")
    ap.add_argument("--ablation", action="store_true",
                    help="also report leave-one-out consensus.")
    args = ap.parse_args()

    cfg = PipelineConfig()
    if args.experts:
        active = [get(eid.strip()) for eid in args.experts.split(",") if eid.strip()]
    else:
        active = get_active(cfg)

    if not os.path.exists(args.master):
        sys.exit(f"master CSV not found: {args.master}")
    if not os.path.exists(args.test):
        sys.exit(f"test split not found: {args.test}")

    rows = _load_test_rows(args.master, args.test)
    if len(rows) == 0:
        sys.exit("No test rows after joining split keys to master CSV. Has the "
                 "test split been built and the master CSV generated?")

    print(f"Loaded {len(rows)} sealed test rows "
          f"(directional Target_Label) from {args.master}\n"
          f"  that are also in {args.test}")
    per_expert_eval(rows, active)
    consensus_eval(rows, active, args.ablation)

    print("\nReminder: this scores the EXPERTS directly on the sealed test set. "
          "An expert that doesn't clear ~50% / mcc>0 here will not give the "
          "student edge no matter how well it's distilled.")


if __name__ == "__main__":
    main()
