"""
Example cross-sectional expert: 5-day return rank vs the rest of the universe.

Not in the default active set, it's here as a template for an expert that
compares each stock against all the others on the same date rather than its
own history.

build_features groups the panel by Date and ranks each stock's 5-day return
against its peers, giving one new column XSect_Mom_Rank (0..1 percentile).
Once that's merged in it's just a normal column, so make_input() and signal()
work the same as for the single ticker experts.

To use it:
    1. import experts.examples.sector_relative_momentum   # registers "xsmom"
    2. add "xsmom" to config.active_experts
    3. run build_features over the panel to get XSect_Mom_Rank, then score it:
       python train/eval_experts.py --experts xsmom

It's deterministic so it can be scored on the test set without any LLM/ GPU cost.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

if __package__ in (None, ""):  # allow `python experts/examples/sector_relative_momentum.py`
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experts.base import DeterministicExpert
from experts.registry import register

RANK_COL = "XSect_Mom_Rank"


class SectorRelativeMomentumExpert(DeterministicExpert):
    id = "xsmom"
    name = "Cross-Sectional Momentum"
    source_tag = "XSECT_MOMENTUM"
    prompt_title = "Cross-Sectional Momentum Expert"
    signal_column = "XSect_Mom_Signal"
    input_column = "XSect_Mom_Input"
    view_column = "XSect_Mom_View"
    include_basis = True
    data_scope = "cross_sectional"
    teacher_description = "5-day return rank vs the universe on each date"

    # ------------------------------------------------------------------
    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame:
        """For each Date rank every stock's 5-day return against the others
        that day. Returns (Ticker, Date, XSect_Mom_Rank)."""
        df = panel[["Ticker", "Date", "Stock_5d_Return"]].copy()
        # percentile rank within each date's cross-section (0..1, 0.5 = median)
        df[RANK_COL] = (
            df.groupby("Date")["Stock_5d_Return"].rank(pct=True, method="average")
        )
        return df[["Ticker", "Date", RANK_COL]]

    # ------------------------------------------------------------------
    def make_input(self, row) -> str:
        rank = row.get(RANK_COL)
        rank_str = "N/A" if pd.isna(rank) else f"{rank * 100:.0f}th percentile"
        return f"""ASSET: {row['Ticker']}
DATE: {row['Date'].date()}

[CROSS-SECTIONAL MOMENTUM]
- 5-Day Return Rank vs universe: {rank_str}
  (above the 50th percentile = stronger 5-day momentum than the median peer)"""

    def compute(self, row):
        """Top half -> BUY, bottom half -> SELL. Missing rank defaults to SELL,
        same as the price/macro expert."""
        rank = row.get(RANK_COL)
        if pd.isna(rank):
            label = "SELL"
            view = f"{label}: cross-sectional rank unavailable; default SELL."
            return label, view
        label = "BUY" if rank >= 0.5 else "SELL"
        view = (f"{label}: 5-day return at the {rank * 100:.0f}th percentile of "
                f"the universe ({'above' if rank >= 0.5 else 'below'} median).")
        return label, view


register(SectorRelativeMomentumExpert())


# ----------------------------------------------------------------------
# quick demo/ sanity check, run with:
#   python experts/examples/sector_relative_momentum.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    panel = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
        "Date": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 3),
        "Stock_5d_Return": [0.01, 0.05, -0.02, -0.03, 0.00, 0.04],
    })
    e = SectorRelativeMomentumExpert()
    feats = e.build_features(panel, None)
    merged = panel.merge(feats, on=["Ticker", "Date"])
    print(merged.to_string(index=False))
    for _, r in merged.iterrows():
        print(f"  {r['Ticker']} {r['Date'].date()}: signal={e.signal(r)} | "
              f"{e.compute(r)[1]}")
    # on 2025-01-02 BBB (+0.05) is top -> BUY, CCC (-0.02) is bottom -> SELL
    top = merged[(merged.Date == "2025-01-02") & (merged.Ticker == "BBB")].iloc[0]
    bot = merged[(merged.Date == "2025-01-02") & (merged.Ticker == "CCC")].iloc[0]
    assert e.signal(top) == "UP" and e.signal(bot) == "DOWN"
    print("\n[OK] cross-sectional build_features + signal demo passed.")
