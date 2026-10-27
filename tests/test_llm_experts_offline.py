"""
Offline tests for the sentiment and fundamental LLM experts (no LLM/ GPU).

Checks Sent_Input/ Fund_Input and Fund_View match the stored master CSV (they
only depend on non-LLM columns or the stored signal), aggregate_sentiment
(abstain, exact zero -> BUY, clamping) and parse_response incl. schema errors.
Sent_Signal/ Fund_Signal themselves come from Gemini so can't be checked here,
that needs a batch API run.
"""
import math
import os
import sys

import pandas as pd

try:
    import pytest
except ImportError:
    class _PytestShim:
        class _Skip(Exception):
            pass

        def skip(self, msg):
            raise self._Skip(msg)
    pytest = _PytestShim()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experts.base import ResponseSchemaError                      # noqa: E402
from experts.sentiment import SentimentExpert, aggregate_sentiment  # noqa: E402
from experts.fundamental import (                                  # noqa: E402
    FundamentalExpert, fundamental_view,
)

MASTER_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "master_training_data.csv",
)

SN = SentimentExpert()
FN = FundamentalExpert()


def _load(col):
    if not os.path.exists(MASTER_CSV):
        pytest.skip(f"master CSV not found at {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df[col].notna()].copy()
    if len(df) == 0:
        pytest.skip(f"no rows with a populated {col}")
    return df


# ------------------------------------------------------------------
# deterministic prompt blocks vs stored master CSV
# ------------------------------------------------------------------
def test_sentiment_input_byte_identical():
    df = _load("Sent_Input")
    bad = [(r["Ticker"], r["Date"]) for _, r in df.iterrows()
           if SN.make_input(r).strip() != r["Sent_Input"]]
    assert not bad, f"{len(bad)} Sent_Input mismatches (first: {bad[:3]})"
    print(f"\n[OK] {len(df):,} rows: Sent_Input byte-identical.")


def test_fundamental_input_byte_identical():
    df = _load("Fund_Input")
    bad = [(r["Ticker"], r["Date"]) for _, r in df.iterrows()
           if FN.make_input(r).strip() != r["Fund_Input"]]
    assert not bad, f"{len(bad)} Fund_Input mismatches (first: {bad[:3]})"
    print(f"[OK] {len(df):,} rows: Fund_Input byte-identical.")


def test_fundamental_view_byte_identical():
    """Fund_View == fundamental_view(Fund_Signal) on every populated row."""
    df = _load("Fund_View")
    bad = [(r["Ticker"], r["Date"]) for _, r in df.iterrows()
           if fundamental_view(r["Fund_Signal"]) != r["Fund_View"]]
    assert not bad, f"{len(bad)} Fund_View mismatches (first: {bad[:3]})"
    print(f"[OK] {len(df):,} rows: Fund_View byte-identical.")


# ------------------------------------------------------------------
# aggregate_sentiment
# ------------------------------------------------------------------
def test_aggregate_sentiment_abstains_on_empty():
    label, view, agg = aggregate_sentiment([])
    assert label == "NEUTRAL"
    assert agg == 0.0
    assert "abstains" in view


def test_aggregate_sentiment_exact_zero_is_buy():
    # mean of exactly 0.0 -> BUY (paper uses >=)
    label, view, agg = aggregate_sentiment([-0.5, 0.5])
    assert label == "BUY"
    assert agg == 0.0


def test_aggregate_sentiment_sign_and_clamp():
    label, view, agg = aggregate_sentiment([-0.4, -0.8])
    assert label == "SELL"
    assert agg == -0.6
    # out of range scores get clamped to [-1, 1]
    label2, _, agg2 = aggregate_sentiment([5.0, 5.0])
    assert label2 == "BUY" and agg2 == 1.0
    # non-numeric entries are skipped
    label3, _, agg3 = aggregate_sentiment([None, "x", 0.2])
    assert label3 == "BUY" and math.isclose(agg3, 0.2)


def test_aggregate_singular_plural_view():
    _, view1, _ = aggregate_sentiment([0.2])
    assert "mean of 1 headline impact score)" in view1  # singular
    _, view2, _ = aggregate_sentiment([0.2, 0.2])
    assert "mean of 2 headline impact scores)" in view2  # plural


# ------------------------------------------------------------------
# parse_response
# ------------------------------------------------------------------
def test_sentiment_parse_response_matches_aggregate():
    result = {"scores": [{"headline": "a", "score": 0.3},
                         {"headline": "b", "score": -0.1}]}
    got = SN.parse_response(result)
    want = aggregate_sentiment([0.3, -0.1])
    assert got == want


def test_sentiment_parse_response_schema_failure():
    try:
        SN.parse_response({"not_scores": []})
        assert False, "expected ResponseSchemaError"
    except ResponseSchemaError:
        pass


def test_fundamental_parse_response():
    assert FN.parse_response({"signal": "BUY"}) == (
        "BUY", fundamental_view("BUY"), None)
    assert FN.parse_response({"signal": "SELL"}) == (
        "SELL", fundamental_view("SELL"), None)


def test_fundamental_parse_response_schema_failure():
    try:
        FN.parse_response({"no_signal": "x"})
        assert False, "expected ResponseSchemaError"
    except ResponseSchemaError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            cls = type(e).__name__
            if cls == "_Skip" or "Skip" in cls:
                print(f"SKIP {fn.__name__}: {e}")
            else:
                print(f"FAIL {fn.__name__}: {e}")
                raise
    print("\nAll offline LLM-expert checks passed.")
