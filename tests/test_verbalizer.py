"""
Tests for core.verbalizer.extract_direction, the 8 cases are from
reference_results/BASELINE.md section 2.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.verbalizer import extract_direction, DTW_UP_WORDS, DTW_DOWN_WORDS


# ----------------------------------------------------------------------------
# cases from BASELINE.md section 2
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    # ticker abbreviations with periods must not confuse the scanner
    ("D.R. Horton is expected to rise",                       "UP"),
    ("A.O. Smith is expected to decline sharply",             "DOWN"),
    # negation: "unlikely to decline" -> UP
    ("The stock is unlikely to decline and should rise",      "UP"),
    # soft negator: "little downside" -> UP
    ("There is little downside from here",                    "UP"),
    # soft negator: "limited upside" -> DOWN
    ("We see limited upside in the near term",                "DOWN"),
    # earlier claim wins: "rise" before "decline"
    ("The stock will rise, though some decline is possible",  "UP"),
    # descriptive noun phrase: "downward movement"
    ("The price is expected to see a downward movement",      "DOWN"),
    # no direction in the first SCAN_CAP=25 tokens
    (
        "The macroeconomic environment remains uncertain with "
        "cross-currents from interest rate policy, currency volatility, "
        "and geopolitical tension. Analysts are watching closely.",
        None,
    ),
])
def test_extract_direction(text, expected):
    assert extract_direction(text) == expected


# ----------------------------------------------------------------------------
# DTW word lists, Ploutos appendix C has exactly 4 words per direction
# ----------------------------------------------------------------------------
def test_dtw_word_list_sizes():
    assert len(DTW_UP_WORDS) == 4, f"DTW_UP_WORDS expanded: {DTW_UP_WORDS}"
    assert len(DTW_DOWN_WORDS) == 4, f"DTW_DOWN_WORDS expanded: {DTW_DOWN_WORDS}"


def test_dtw_word_lists_unchanged():
    assert DTW_UP_WORDS   == ["up", "boost", "positive", "rise"]
    assert DTW_DOWN_WORDS == ["down", "reduce", "negative", "drop"]


# ----------------------------------------------------------------------------
# leading "Prediction:" label gets stripped
# ----------------------------------------------------------------------------
def test_strips_prediction_label():
    assert extract_direction("Prediction: The stock should rise.") == "UP"
    assert extract_direction("PREDICTION: decline expected") == "DOWN"


# ----------------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------------
def test_empty_and_none_input():
    assert extract_direction("") is None
    assert extract_direction("   ") is None


def test_fallback_to_full():
    # "rise" only appears well past SCAN_CAP
    filler = " ".join(["context"] * 30)
    text = filler + " rise"
    assert extract_direction(text) is None
    assert extract_direction(text, fallback_to_full=True) == "UP"
