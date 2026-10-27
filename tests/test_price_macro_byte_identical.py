"""
Checks the price/macro expert reproduces the stored values exactly.

pm signal, view and input block are deterministic, so recompute them with
PriceMacroExpert and compare to data/master_training_data.csv. Runs offline,
skips if the master CSV isn't there.
"""
import os
import sys

import pandas as pd

try:
    import pytest
except ImportError:  # allow running as a plain script without pytest installed
    class _PytestShim:
        class _Skip(Exception):
            pass

        def skip(self, msg):
            raise self._Skip(msg)
    pytest = _PytestShim()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experts.price_macro import PriceMacroExpert  # noqa: E402

MASTER_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "master_training_data.csv",
)

EXPERT = PriceMacroExpert()


def _load_pm_rows():
    if not os.path.exists(MASTER_CSV):
        pytest.skip(f"master CSV not found at {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Price_Macro_Signal"].notna()].copy()
    if len(df) == 0:
        pytest.skip("no rows with a populated Price_Macro_Signal")
    return df


def test_price_macro_signal_view_input_byte_identical():
    df = _load_pm_rows()
    mismatches = {"signal": [], "view": [], "input": []}

    for _, row in df.iterrows():
        label, view = EXPERT.compute(row)
        if label != row["Price_Macro_Signal"]:
            mismatches["signal"].append((row["Ticker"], row["Date"]))
        if view != row["Price_Macro_View"]:
            mismatches["view"].append((row["Ticker"], row["Date"]))
        if EXPERT.make_input(row).strip() != row["Price_Macro_Input"]:
            mismatches["input"].append((row["Ticker"], row["Date"]))

    for kind, bad in mismatches.items():
        assert not bad, (
            f"{len(bad)} {kind} mismatches vs stored master CSV "
            f"(first few: {bad[:3]})"
        )

    print(f"\n[OK] {len(df):,} rows: signal/view/input byte-identical to "
          f"pre-refactor master CSV.")


def test_signal_normalizes_to_up_down():
    """signal() gives UP/DOWN matching the stored BUY/SELL on every row."""
    df = _load_pm_rows()
    norm = {"BUY": "UP", "SELL": "DOWN", "NEUTRAL": "NEUTRAL"}
    bad = []
    for _, row in df.iterrows():
        got = EXPERT.signal(row)
        want = norm.get(row["Price_Macro_Signal"])
        if got != want:
            bad.append((row["Ticker"], row["Date"], got, want))
    assert not bad, f"{len(bad)} normalization mismatches (first: {bad[:3]})"


if __name__ == "__main__":
    test_price_macro_signal_view_input_byte_identical()
    test_signal_normalizes_to_up_down()
    print("All price/macro byte-identity checks passed.")
