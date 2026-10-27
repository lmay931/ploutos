# accuracy/ MCC helpers for the eval scripts

import math


def mcc(tp, tn, fp, fn):
    """Matthews correlation coefficient, 0.0 if the denominator is 0."""
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den else 0.0


def confusion(pairs):
    """(tp, tn, fp, fn) with UP as the positive class. pairs is (pred, truth)."""
    tp = sum(1 for p, t in pairs if p == "UP" and t == "UP")
    tn = sum(1 for p, t in pairs if p == "DOWN" and t == "DOWN")
    fp = sum(1 for p, t in pairs if p == "UP" and t == "DOWN")
    fn = sum(1 for p, t in pairs if p == "DOWN" and t == "UP")
    return tp, tn, fp, fn


def report_block(name, pairs):
    """Print name, n, accuracy and MCC on one line."""
    if not pairs:
        print(f"  {name:<28} (no rows)")
        return
    tp, tn, fp, fn = confusion(pairs)
    n = len(pairs)
    acc = (tp + tn) / n
    m = mcc(tp, tn, fp, fn)
    print(f"  {name:<28} n={n:<5} acc={acc:5.1%}  mcc={m:+.3f}")
