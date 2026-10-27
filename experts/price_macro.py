"""
Price/macro expert (deterministic, no LLM).

Combines Price_vs_MA20, Stock_5d_Return and Relative_Momentum with a fixed rule.
An LLM didn't add anything here and its label and explanation sometimes didn't
match, so the label and view are computed in code.

Data scope: single ticker. The columns it uses are actually built upstream in
build_master_training_dataset.py, so build_features here just selects them.
A new deterministic expert would compute its features in build_features.
"""
from __future__ import annotations

import pandas as pd

from experts.base import DeterministicExpert
from experts.formatting import fmt
from experts.registry import register

# columns read from each row
REQUIRED_COLUMNS = [
    "Ticker", "Date", "Close", "Volume",
    "Price_vs_MA20", "Stock_5d_Return", "Relative_Momentum",
    "SPY_5d_Return", "SPY_Vol_20d",
]


def _num(val):
    """Convert a possibly missing value to float, NaN -> 0.0.

    The rule only looks at signs, so treating missing as 0 (non-positive) keeps
    it simple. Rows where everything is missing end up as SELL.
    """
    return 0.0 if pd.isna(val) else float(val)


def build_price_macro_input(row):
    vol = row.get('Volume')
    volume_str = f"{int(vol)}" if not pd.isna(vol) else 'N/A'
    return f"""ASSET: {row['Ticker']}
DATE: {row['Date'].date()}

[PRICE ACTION]
- Current Close: ${row['Close']:.2f}
- Price vs 20-Day MA: {fmt(row.get('Price_vs_MA20'), is_pct=True)}
- 5-Day Trend Return: {fmt(row.get('Stock_5d_Return'), is_pct=True)}
- Volume: {volume_str}

[MACRO CONTEXT]
- SPY 5-Day Return: {fmt(row.get('SPY_5d_Return'), is_pct=True)}
- Stock vs SPY (Relative Momentum): {fmt(row.get('Relative_Momentum'), is_pct=True)}
- Market Volatility Regime (SPY 20d Ann. Vol): {fmt(row.get('SPY_Vol_20d'), is_pct=True)}"""


def price_label(ma_dev, trend_5d, rel_mom):
    """Rule, in order:
      1. all three agree -> that direction
      2. mixed -> Relative Momentum decides (>0 BUY, <0 SELL)
      3. Relative Momentum exactly 0 -> tiebreak on Price vs 20-Day MA
    """
    votes = [ma_dev > 0, trend_5d > 0, rel_mom > 0]
    if all(votes):
        return "BUY"
    if not any(votes):
        return "SELL"
    if rel_mom > 0:
        return "BUY"
    if rel_mom < 0:
        return "SELL"
    return "BUY" if ma_dev > 0 else "SELL"  # rel_mom == 0 tiebreak


def price_view(label, ma_dev, trend_5d, rel_mom):
    p_ma, p_tr, p_mom = (fmt(ma_dev, True), fmt(trend_5d, True), fmt(rel_mom, True))
    agree = (ma_dev > 0) == (trend_5d > 0) == (rel_mom > 0)
    if agree:
        basis = ("all three signals are "
                 f"{'bullish' if label == 'BUY' else 'bearish'}")
    elif rel_mom != 0:
        basis = f"signals diverge and Relative Momentum ({p_mom}) is decisive"
    else:
        basis = ("signals diverge with flat Relative Momentum, so Price vs "
                 f"20-Day MA ({p_ma}) is the tiebreak")
    return (f"{label}: Price vs 20-Day MA {p_ma}, 5-Day Trend Return {p_tr}, "
            f"Relative Momentum {p_mom} — {basis}.")


def compute_price_macro(row):
    """Return (label, view) for the price/macro expert."""
    ma_dev = _num(row.get('Price_vs_MA20'))
    trend_5d = _num(row.get('Stock_5d_Return'))
    rel_mom = _num(row.get('Relative_Momentum'))
    label = price_label(ma_dev, trend_5d, rel_mom)
    return label, price_view(label, ma_dev, trend_5d, rel_mom)


class PriceMacroExpert(DeterministicExpert):
    id = "pm"
    name = "Price & Macro"
    source_tag = "PRICE_MACRO"
    prompt_title = "Price & Macro Expert"
    signal_column = "Price_Macro_Signal"
    input_column = "Price_Macro_Input"
    view_column = "Price_Macro_View"
    include_basis = True          # teacher block prints "Basis: {view}"
    data_scope = "single_ticker"
    teacher_description = "price action and macro/relative-momentum signals"

    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame:
        """Just selects the columns this expert uses, they're built upstream in
        build_master_training_dataset.py."""
        cols = [c for c in REQUIRED_COLUMNS if c in panel.columns]
        return panel[cols].copy()

    def make_input(self, row) -> str:
        return build_price_macro_input(row)

    def compute(self, row):
        return compute_price_macro(row)


register(PriceMacroExpert())
