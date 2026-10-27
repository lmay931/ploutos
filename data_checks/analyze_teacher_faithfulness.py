#!/usr/bin/env python3
"""
Rough faithfulness check on the rearview SFT data from teacher_model_batch.py.

There's no way to know if a rationale is actually "true", so this looks at a
proxy instead: on rows where the experts were wrong (low experts_correct), does
the teacher's main argument rely on generic features (distance from the 20-day
MA, P/E, stale data) that could be argued either way once you know the answer?
If the same feature gets used for DOWN on overextended winners and UP on
oversold losers, the teacher is probably just making up a story to fit the
label, and the student would be learning rhetoric rather than anything useful.

Nothing gets dropped, it only reports and optionally dumps the flagged rows to
check by hand. If the numbers look bad, DROP_DIRECTION_MISMATCHES in
teacher_model_batch.py (or downsampling ec=0) is the place to act on it.

Usage:
    python analyze_teacher_faithfulness.py data/sft_training_data.jsonl
    python analyze_teacher_faithfulness.py data/sft_training_data.jsonl --dump flagged.jsonl
    python analyze_teacher_faithfulness.py data/sft_training_data.jsonl --max-ec 1
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# generic "hindsight" phrases the teacher tends to use when it knows the answer
# but the experts don't back it up, grouped into families. using them isn't
# wrong in itself - the problem is when the same family is used to justify both
# directions across the data.
# patterns are kept broad on purpose so expect false positives, check the
# dumped rows before trusting any one flag
# ---------------------------------------------------------------------------
GENERIC_FAMILIES = {
    "ma_extension": [
        r"overextend", r"mean[- ]?revert", r"mean[- ]?reversion",
        r"above its 20-day", r"below its 20-day", r"oversold", r"overbought",
        r"technical exhaustion", r"profit[- ]?taking", r"relief rally",
        r"reclaim", r"pull ?back to", r"snap back",
    ],
    "valuation_pe": [
        r"\bP/E\b", r"price-to-earnings", r"stretched valuation",
        r"premium valuation", r"overvalued", r"undervalued",
        r"valuation (?:reset|correction|ceiling|floor|contraction)",
        r"trading at a (?:premium|discount)",
    ],
    "data_staleness": [
        r"\d{2,3}\s*days?\s*old", r"stale", r"may not (?:reflect|account for)",
        r"lagging behind the company", r"data is .*old",
    ],
    "volatility_regime": [
        r"high(?:er)? (?:market )?volatility", r"volatility regime",
        r"systemic risk", r"broad(?:er)? market (?:weakness|downturn|decline)",
        r"macro(?:economic)? (?:headwind|pressure|volatility)",
    ],
}

COMPILED = {
    fam: [re.compile(p, re.IGNORECASE) for p in pats]
    for fam, pats in GENERIC_FAMILIES.items()
}

# words that point to something specific to the company (a news event or a hard
# fundamental), if they show up in the prediction the row is less suspicious
GROUNDED_HINTS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"FDA", r"approval", r"trial", r"earnings (?:beat|miss|surpass|decline)",
        r"price target", r"upgrade", r"downgrade", r"acquisition", r"merger",
        r"lawsuit", r"strike", r"layoff", r"contract", r"buyback",
        r"dividend", r"guidance", r"recall", r"settlement", r"partnership",
        r"free cash flow", r"negative free cash flow", r"current ratio",
        r"liquidity", r"loss-making",
    ]
]


def first_sentence(text: str) -> str:
    """First sentence of the prediction - it has the direction (what extract_direction reads) and usually the main argument."""
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


def families_hit(text: str) -> set:
    hits = set()
    for fam, regexes in COMPILED.items():
        if any(r.search(text) for r in regexes):
            hits.add(fam)
    return hits


def has_grounded_hint(text: str) -> bool:
    return any(r.search(text) for r in GROUNDED_HINTS)


def get_prediction_text(record: dict) -> str:
    """Gets the prediction text from the model/ assistant message, or the whole answer if there's no 'Prediction:' block."""
    msgs = record.get("messages", [])
    answer = ""
    for m in msgs:
        if m.get("role") in ("model", "assistant"):
            answer = m.get("content", "")
    if not answer:
        return ""
    m = re.search(r"Prediction:\s*(.+)\Z", answer, re.IGNORECASE | re.DOTALL)
    return (m.group(1) if m else answer).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to teacher SFT JSONL")
    ap.add_argument("--max-ec", type=int, default=1,
                    help="Analyze rows with experts_correct <= this as the "
                         "high-risk set (default 1). ec=0 is highest risk.")
    ap.add_argument("--dump", default=None,
                    help="Optional path to write flagged high-risk rows for "
                         "hand audit.")
    args = ap.parse_args()

    rows = []
    with open(args.jsonl) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] line {ln}: bad JSON ({e}); skipped",
                      file=sys.stderr)

    if not rows:
        print("No rows parsed. Is the path right and the file non-empty?")
        return

    # across the whole file, count how often each family is used for UP vs DOWN
    family_by_direction = defaultdict(lambda: Counter())  # family -> {UP, DOWN}
    total = len(rows)
    ec_dist = Counter()
    high_risk = []          # rows with ec <= max-ec
    flagged = []            # high-risk rows whose decisive sentence is generic-only

    for r in rows:
        meta = r.get("metadata", {})
        ec = meta.get("experts_correct")
        truth = (meta.get("ground_truth") or "").upper()
        ec_dist[ec] += 1

        pred = get_prediction_text(r)
        fs = first_sentence(pred)
        fams_first = families_hit(fs)
        fams_all = families_hit(pred)

        for fam in fams_all:
            if truth in ("UP", "DOWN"):
                family_by_direction[fam][truth] += 1

        if ec is not None and ec <= args.max_ec:
            grounded = has_grounded_hint(pred)
            # most suspicious case: the first sentence uses a generic family and
            # nothing in the prediction is tied to the actual inputs
            generic_only = bool(fams_first) and not grounded
            row_record = {
                "ticker": meta.get("ticker"),
                "date": meta.get("date"),
                "ground_truth": truth,
                "experts_correct": ec,
                "experts_directional": meta.get("experts_directional"),
                "first_sentence_families": sorted(fams_first),
                "all_families": sorted(fams_all),
                "has_grounded_anchor": grounded,
                "generic_only_decisive": generic_only,
                "first_sentence": fs,
            }
            high_risk.append(row_record)
            if generic_only:
                flagged.append({**row_record, "full_prediction": pred,
                                "_raw": r})

    # report
    print("=" * 70)
    print(f"Teacher faithfulness screen - {total} rows")
    print("=" * 70)

    print("\nexperts_correct distribution:")
    for ec in sorted(ec_dist, key=lambda x: (x is None, x)):
        bar = "#" * min(50, ec_dist[ec])
        print(f"  ec={ec!s:>4}: {ec_dist[ec]:>4}  {bar}")

    print("\n--- Confabulation tell: does each generic family argue BOTH ways? ---")
    print("(A family used heavily for both UP and DOWN is a direction-conditioned")
    print(" template, not grounded inference. Balanced split = strongest tell.)\n")
    print(f"  {'family':<18}{'->DOWN':>8}{'->UP':>8}{'balance':>10}")
    for fam in GENERIC_FAMILIES:
        d = family_by_direction[fam]["DOWN"]
        u = family_by_direction[fam]["UP"]
        tot = d + u
        if tot == 0:
            bal = "-"
        else:
            # 0.5 = used equally for both directions (worst), 0 = only one direction
            bal = f"{min(d, u) / tot:.2f}"
        print(f"  {fam:<18}{d:>8}{u:>8}{bal:>10}")
    print("\n  balance ~0.50  => family fires in both directions ~equally")
    print("                   (classic hindsight template)")
    print("  balance ~0.00  => family is one-directional (less suspicious)")

    n_hr = len(high_risk)
    n_flag = len(flagged)
    print(f"\n--- High-risk set (experts_correct <= {args.max_ec}) ---")
    print(f"  rows in high-risk set : {n_hr}")
    if n_hr:
        gen_only = sum(1 for r in high_risk if r["generic_only_decisive"])
        no_anchor = sum(1 for r in high_risk if not r["has_grounded_anchor"])
        print(f"  decisive sentence is generic-template AND no grounded anchor:")
        print(f"      {gen_only:>4} / {n_hr}  ({gen_only/n_hr:.0%})  <-- confabulation-suspect")
        print(f"  no grounded anchor anywhere in prediction:")
        print(f"      {no_anchor:>4} / {n_hr}  ({no_anchor/n_hr:.0%})")

    # ec=0 separately since the stratified sampling oversamples these
    ec0 = [r for r in high_risk if r["experts_correct"] == 0]
    if ec0:
        ec0_flag = sum(1 for r in ec0 if r["generic_only_decisive"])
        print(f"\n  ec=0 only ({len(ec0)} rows, the OVERSAMPLED stratum):")
        print(f"      generic-only decisive: {ec0_flag}/{len(ec0)} "
              f"({ec0_flag/len(ec0):.0%})")
        print("      ^ if this is high, your most-weighted training rows are")
        print("        the most likely to be teaching post-hoc rhetoric.")

    if flagged:
        print(f"\n--- {min(8, n_flag)} sample flagged rows (first sentence) ---")
        for r in flagged[:8]:
            print(f"  [{r['ticker']} {r['date']}] truth={r['ground_truth']} "
                  f"ec={r['experts_correct']} fam={r['first_sentence_families']}")
            print(f"      {r['first_sentence']}")

    if args.dump and flagged:
        with open(args.dump, "w") as f:
            for r in flagged:
                out = {k: v for k, v in r.items() if k != "_raw"}
                f.write(json.dumps(out) + "\n")
        print(f"\n[dump] {n_flag} flagged rows -> {args.dump}")

    print("\n" + "=" * 70)
    print("Reminder: flags are a SCREEN, not a verdict. A flagged row can still")
    print("be a fine training example. Audit before dropping/downsampling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
