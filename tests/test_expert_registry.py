"""
Tests for the expert registry/ interface. No data or LLM needed.

Checks pm/sn/fn are registered and resolved from config, each expert has the
Expert attributes, the teacher prompt block matches a golden string on a
synthetic row, and the cross-sectional example expert ranks across tickers.
google.genai and core.api_client are stubbed so the teacher module imports
without the Gemini SDK.
"""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# stub the gemini sdk + api client so teacher_model_batch imports offline
if "google.genai" not in sys.modules:
    g = types.ModuleType("google")
    gg = types.ModuleType("google.genai")
    gg.types = types.ModuleType("google.genai.types")
    sys.modules["google"] = g
    sys.modules["google.genai"] = gg
    sys.modules["google.genai.types"] = gg.types
if "core.api_client" not in sys.modules:
    api = types.ModuleType("core.api_client")
    api.gemini_client = object()
    sys.modules["core.api_client"] = api

from config import PipelineConfig                       # noqa: E402
import experts                                          # noqa: E402,F401
from experts.registry import EXPERT_REGISTRY, get_active, get  # noqa: E402


def test_default_experts_registered():
    assert set(EXPERT_REGISTRY) >= {"pm", "sn", "fn"}
    active = get_active(PipelineConfig())
    assert [e.id for e in active] == ["pm", "sn", "fn"]


def test_interface_surface():
    for e in get_active(PipelineConfig()):
        for attr in ("id", "name", "source_tag", "prompt_title", "signal_column",
                     "input_column", "is_llm", "data_scope"):
            assert getattr(e, attr) not in (None, ""), f"{e.id} missing {attr}"
        assert callable(e.make_input) and callable(e.signal)
        assert e.is_llm == (e.persona is not None)


def test_signal_normalization_round_trip():
    pm = get("pm")
    # BUY/SELL map to UP/DOWN, NEUTRAL and None pass through
    assert pm._normalize("BUY") == "UP"
    assert pm._normalize("SELL") == "DOWN"
    assert pm._normalize("NEUTRAL") == "NEUTRAL"
    assert pm._normalize(None) is None


def test_build_data_context_golden():
    """Teacher data block built from the registry matches the expected text exactly."""
    import teacher_model_batch as t
    row = pd.Series({
        "Price_Macro_Input": "PM_INPUT",
        "Price_Macro_Signal": "BUY",
        "Price_Macro_View": "PM_VIEW",
        "Sent_Input": "SN_INPUT",
        "Sent_Signal": "NEUTRAL",
        "Fund_Input": "FN_INPUT",
        "Fund_Signal": "SELL",
    })
    expected = (
        "[Price & Macro Expert Input]:\n"
        "PM_INPUT\n\n"
        "[Price & Macro Expert Prediction]: BUY\n"
        "Basis: PM_VIEW\n\n"
        "[Sentiment Expert Input]:\n"
        "SN_INPUT\n\n"
        "[Sentiment Expert Prediction]: NEUTRAL\n\n"
        "[Fundamental Expert Input]:\n"
        "FN_INPUT\n\n"
        "[Fundamental Expert Prediction]: SELL"
    )
    assert t.build_data_context(row) == expected


def test_schema_and_source_clause_registry_driven():
    import teacher_model_batch as t
    enum = t.STRICT_SCHEMA["properties"]["bullish_rationales"]["items"][
        "properties"]["source"]["enum"]
    assert enum == ["PRICE_MACRO", "SENTIMENT", "FUNDAMENTAL"]
    assert "__SOURCE_CLAUSE__" not in t.TEACHER_PERSONA
    assert "PRICE_MACRO (price action" in t.TEACHER_PERSONA


def test_cross_sectional_example_ranks_universe():
    from experts.examples.sector_relative_momentum import (
        SectorRelativeMomentumExpert, RANK_COL,
    )
    panel = pd.DataFrame({
        "Ticker": ["A", "B", "C"],
        "Date": pd.to_datetime(["2025-01-02"] * 3),
        "Stock_5d_Return": [0.01, 0.05, -0.02],
    })
    e = SectorRelativeMomentumExpert()
    feats = e.build_features(panel, None).set_index("Ticker")
    assert feats.loc["B", RANK_COL] == 1.0       # top performer
    assert feats.loc["C", RANK_COL] < 0.5        # bottom -> SELL/DOWN
    merged = panel.merge(e.build_features(panel, None), on=["Ticker", "Date"])
    sigs = {r["Ticker"]: e.signal(r) for _, r in merged.iterrows()}
    assert sigs["B"] == "UP" and sigs["C"] == "DOWN"
    # example only registers itself when its module is imported (done above)
    assert "xsmom" in EXPERT_REGISTRY


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("\nAll expert-registry architecture checks passed.")
