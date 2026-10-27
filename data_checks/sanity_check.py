"""
Sanity check on the three expert signals, run after generating them and before
the teacher. Doesn't write anything.

Flags experts that are nearly all one class (not much information) or that are
no better than chance against Target_Label (the teacher will just learn to
ignore them, better to find out now). Also prints coverage and how often the
experts agree with each other.
"""

import pandas as pd
import argparse

SILOS = {
    "Price/Macro":  "Price_Macro_Signal",
    "Sentiment":    "Sent_Signal",
    "Fundamental":  "Fund_Signal",
}

# flag thresholds
CLASS_IMBALANCE_FLAG = 0.65   # >65% one class -> warn
EDGE_FLAG = 0.52              # accuracy <= this -> warn (near chance)


def _label_to_dir(series):
    """Map BUY/SELL (any case) to +1/-1; everything else -> NaN."""
    s = series.astype(str).str.strip().str.upper()
    return s.map({"BUY": 1, "SELL": -1})


def _target_to_dir(series):
    """Map UP/DOWN to +1/-1; FLAT and anything else -> NaN (excluded)."""
    s = series.astype(str).str.strip().str.upper()
    return s.map({"UP": 1, "DOWN": -1})


def main(csv_path):
    df = pd.read_csv(csv_path)
    n_total = len(df)
    print(f"\nLoaded {n_total:,} rows from {csv_path}\n")

    if "Target_Label" not in df.columns:
        raise SystemExit("No Target_Label column found.")

    tgt = _target_to_dir(df["Target_Label"])
    n_directional = int(tgt.notna().sum())
    print(f"Target_Label: {n_directional:,} directional (UP/DOWN), "
          f"{n_total - n_directional:,} FLAT/other (excluded from accuracy).\n")

    # base rate of UP - an expert has to beat this to be any use
    base_up = (tgt == 1).sum() / n_directional if n_directional else float("nan")
    print(f"Target base rate: UP={base_up:.1%}, DOWN={1 - base_up:.1%} "
          f"(a constant-UP predictor would score {max(base_up, 1 - base_up):.1%})\n")

    print("=" * 64)
    print("PER-EXPERT REPORT")
    print("=" * 64)

    expert_dirs = {}
    for name, col in SILOS.items():
        print(f"\n[{name}]  ({col})")
        if col not in df.columns:
            print("   MISSING COLUMN - expert not generated. SKIP.")
            continue

        d = _label_to_dir(df[col])
        expert_dirs[name] = d

        n_filled = int(d.notna().sum())
        coverage = n_filled / n_total if n_total else 0.0
        print(f"   Coverage: {n_filled:,}/{n_total:,} rows labeled ({coverage:.1%})")
        if n_filled == 0:
            print("   No labels present. SKIP.")
            continue

        # class balance
        n_buy = int((d == 1).sum())
        n_sell = int((d == -1).sum())
        buy_frac = n_buy / n_filled
        print(f"   Balance:  BUY={n_buy:,} ({buy_frac:.1%})  "
              f"SELL={n_sell:,} ({1 - buy_frac:.1%})")
        dominant = max(buy_frac, 1 - buy_frac)
        if dominant > CLASS_IMBALANCE_FLAG:
            print(f"   CLASS IMBALANCE: {dominant:.1%} one class "
                  f"(> {CLASS_IMBALANCE_FLAG:.0%}). Low information content.")

        # accuracy vs Target_Label on rows where both are UP/DOWN
        both = d.notna() & tgt.notna()
        n_eval = int(both.sum())
        if n_eval == 0:
            print("   No overlap with directional targets - cannot score edge.")
            continue
        acc = float((d[both] == tgt[both]).mean())
        print(f"   Accuracy vs Target: {acc:.1%}  (on {n_eval:,} rows)")
        if acc <= EDGE_FLAG:
            print(f"   NO EDGE: accuracy <= {EDGE_FLAG:.0%}. "
                  f"Near chance - teacher will likely down-weight this expert.")
        # split by what the expert called, in case it's only right on one side
        for lab, val in (("BUY", 1), ("SELL", -1)):
            m = both & (d == val)
            if m.sum():
                a = float((d[m] == tgt[m]).mean())
                print(f"      when expert says {lab:<4}: {a:.1%} correct "
                      f"({int(m.sum()):,} rows)")

    # cross-expert agreement, just for context
    names = [n for n in SILOS if n in expert_dirs]
    if len(names) >= 2:
        print("\n" + "=" * 64)
        print("CROSS-EXPERT AGREEMENT (pairwise, on rows both labeled)")
        print("=" * 64)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = expert_dirs[names[i]], expert_dirs[names[j]]
                m = a.notna() & b.notna()
                if m.sum():
                    agree = float((a[m] == b[m]).mean())
                    print(f"   {names[i]} vs {names[j]}: agree {agree:.1%} "
                          f"({int(m.sum()):,} rows)")
        # how often all three agree
        if len(names) == 3:
            a, b, c = (expert_dirs[n] for n in names)
            m = a.notna() & b.notna() & c.notna()
            if m.sum():
                unan = float(((a[m] == b[m]) & (b[m] == c[m])).mean())
                print(f"   All three unanimous: {unan:.1%} ({int(m.sum()):,} rows)")

    print("\nDone. Review any CLASS IMBALANCE/ NO EDGE lines before training the teacher.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/master_training_data.csv")
    args = parser.parse_args()
    main(args.csv)