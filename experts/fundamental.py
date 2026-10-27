"""
Fundamental expert (LLM, signal only).

Following Ploutos Table 11, the LLM just gives a BUY/SELL signal from the
metrics in the input block and the rationale is left to the aggregator
(PloutosGPT) downstream. Fund_View is a fixed template restating the signal.

Data scope: single ticker, uses the ticker's own fundamentals.
"""
from __future__ import annotations

import pandas as pd

from experts.base import LLMExpert, ResponseSchemaError
from experts.formatting import fmt
from experts.registry import register

# columns read from each row
REQUIRED_COLUMNS = [
    "Ticker", "Date", "PE_Ratio", "eps_ttm", "ROE", "ROE_distorted", "ROA",
    "FCF_Yield", "Gross_Margin", "Current_Ratio", "Is_Financial",
    "fiscalDateEnding", "Filing_Date",
]

# the LLM only returns a signal here
EXPERT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "signal": {
            "type": "STRING",
            "enum": ["BUY", "SELL"],
            "description": "Directional signal based solely on the provided data."
        }
    },
    "required": ["signal"]
}

FUNDAMENTAL_PERSONA = """
You are a Fundamental Financial Analyst.

[CONTEXT]
P/E is computed on trailing-12-month earnings (a stable annual anchor, not
a single annualized quarter).

[READING THE METRICS]
- ROA (TTM net income / assets) is your primary capital-efficiency metric.
- IF P/E shows "N/A (loss-making period)": the company lost money over the
  trailing year - materially bearish for valuation.
- IF ROE shows "N/A (buyback-distorted equity)": ignore ROE and judge
  profitability on ROA. This is not weakness.
- IF metrics show "N/A (not meaningful for financials)": the company is a
  bank/insurer/REIT; judge it on P/E, ROE, and ROA only. The N/As are not
  bearish. For these financial names, ROA is STRUCTURALLY low (often <1.5%)
  and is NOT evidence of poor efficiency — ROE is the relevant efficiency
  metric. Judge a financial's ROE against ~10% as a rough neutral anchor,
  and do NOT call a sub-1.5% ROA "poor" for a bank/insurer/REIT.
- Filing age is shown in the header; treat older data as weaker and more
  uncertain, not as a directional signal.

[HOW TO REASON]
Weigh these dimensions together; no single one is decisive on its own:
  - Valuation: is P/E justified by the growth/quality on display?
  - Capital efficiency: ROA primarily (ROE for financials); ROE when not
    buyback-distorted.
  - Cash generation: FCF yield.
  - Profitability/durability: gross margin.
  - Balance sheet: current ratio (skip for financials).

A loss-making trailing year (P/E N/A loss-making) or negative ROA is a
strong bearish weight, but may be outweighed by strong cash generation or a
clear, durable margin profile.

[DETERMINISM]
Reason such that the SAME data would always yield the SAME signal. Do not let
noise-level differences (e.g. a 0.4% vs 0.6% ROA for a bank) flip the call;
anchor on the dimensions above, not on hair-splitting deltas.

You must commit to BUY or SELL. Output ONLY the signal — no explanation. The
rationale is generated downstream by the aggregator, not by this expert.
"""


def fmt_pe(row):
    pe = row.get('PE_Ratio')
    if not pd.isna(pe):
        return f"{pe:.2f}"
    eps_ttm = row.get('eps_ttm')
    if not pd.isna(eps_ttm) and eps_ttm <= 0:
        return "N/A (loss-making period)"
    return "N/A"


def fmt_roe(row):
    if bool(row.get('ROE_distorted', False)):
        return "N/A (buyback-distorted equity)"
    roe = row.get('ROE')
    if not pd.isna(roe):
        return f"{'+' if roe > 0 else ''}{roe * 100:.2f}%"
    return "N/A"


def fmt_pct_or_financial_na(row, col):
    val = row.get(col)
    if not pd.isna(val):
        return f"{'+' if val > 0 else ''}{val * 100:.2f}%"
    if bool(row.get('Is_Financial', False)):
        return "N/A (not meaningful for financials)"
    return "N/A (not separately reported)"


def fmt_current_ratio(row):
    cr = row.get('Current_Ratio')
    if not pd.isna(cr):
        return f"{cr:.2f}"
    if bool(row.get('Is_Financial', False)):
        return "N/A (not meaningful for financials)"
    return "N/A"


def fmt_filing_age(row):
    fde = pd.to_datetime(row.get('fiscalDateEnding'), errors='coerce')
    dt = pd.to_datetime(row.get('Date'), errors='coerce')
    if pd.isna(fde) or pd.isna(dt):
        fd = row.get('Filing_Date', 'N/A')
        return f"as of {fd}"
    age_days = (dt - fde).days
    return (f"reflects quarter ending {fde.date()}, "
            f"~{age_days} days old as of decision date")


def build_fundamental_input(row):
    return f"""ASSET: {row['Ticker']}
DATE: {row['Date'].date()}

[FUNDAMENTAL DATA - {fmt_filing_age(row)}]
- P/E Ratio (TTM): {fmt_pe(row)}
- Free Cash Flow Yield (TTM): {fmt_pct_or_financial_na(row, 'FCF_Yield')}
- Gross Margin: {fmt_pct_or_financial_na(row, 'Gross_Margin')}
- Return on Assets (TTM ROA): {fmt(row.get('ROA'), is_pct=True)}
- Return on Equity (TTM ROE): {fmt_roe(row)}
- Current Ratio: {fmt_current_ratio(row)}
"""


def fundamental_view(label):
    """Fixed template view for the fundamental expert.

    The expert only returns a signal, so to keep Fund_View in the same shape as
    the other experts (and so it can never contradict the signal) we just
    restate the call. The real rationale comes from the aggregator downstream,
    using Fund_Input and the ground truth.
    """
    return (f"{label}: fundamental call from the metrics in Fund_Input; "
            f"rationale generated downstream by the aggregator.")


class FundamentalExpert(LLMExpert):
    id = "fn"
    name = "Fundamental"
    source_tag = "FUNDAMENTAL"
    prompt_title = "Fundamental Expert"
    signal_column = "Fund_Signal"
    input_column = "Fund_Input"
    view_column = "Fund_View"
    include_basis = False
    data_scope = "single_ticker"
    teacher_description = "valuation, profitability, balance-sheet metrics"
    persona = FUNDAMENTAL_PERSONA
    schema = EXPERT_SCHEMA
    aggregates = False   # LLM gives the signal directly

    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame:
        cols = [c for c in REQUIRED_COLUMNS if c in panel.columns]
        return panel[cols].copy()

    def make_input(self, row) -> str:
        return build_fundamental_input(row)

    def parse_response(self, result: dict):
        """result looks like {"signal": "BUY"|"SELL"}. No aggregate score here.
        Raises ResponseSchemaError if 'signal' is missing."""
        if 'signal' not in result:
            raise ResponseSchemaError("fundamental response missing 'signal'")
        label = result['signal']
        return label, fundamental_view(label), None


register(FundamentalExpert())
