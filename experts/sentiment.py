"""
Sentiment expert (LLM scoring + aggregation in code).

As in Ploutos (Tong et al. 2024, sec 3.1.1) the LLM scores each headline's
price impact in [-1, +1], then the code takes the mean and its sign
(>= 0 -> BUY, < 0 -> SELL). So the LLM never picks BUY/SELL itself. Rows with
no scorable headlines give NEUTRAL (abstain), which only exists at the expert level.

Note the persona contains a literal "{_HORIZON_DAYS}" - it was never .format()-ed
so it gets sent like that. Changing it would change every prompt.

Data scope: single ticker, uses the ticker's own recent headlines.
"""
from __future__ import annotations

import statistics

import pandas as pd

from experts.base import LLMExpert, ResponseSchemaError
from experts.registry import register

# columns read from each row
REQUIRED_COLUMNS = ["Ticker", "Date", "Top_Headlines"]

# the LLM doesn't give a signal here, it just scores each headline's 5-day
# price impact in [-1, +1]. code aggregates and signs it (see parse_response)
SENTIMENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "ARRAY",
            "description": "One object per headline, in the order given.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "headline": {
                        "type": "STRING",
                        "description": "First ~8 words of the headline, verbatim."
                    },
                    "score": {
                        "type": "NUMBER",
                        "description": "Price impact in [-1.0, +1.0]."
                    }
                },
                "required": ["headline", "score"]
            }
        }
    },
    "required": ["scores"]
}

SENTIMENT_PERSONA = """
You are a financial news analyst. You will be given a list of news headlines
related to a single stock, each with a date. For EACH headline, rate its
likely direct impact on THAT stock's price over the next {_HORIZON_DAYS} trading days.

Use a score from -1.0 to +1.0:
  +1.0 = strongly positive for the stock's price
   0.0 = no material price impact / irrelevant / purely informational
  -1.0 = strongly negative for the stock's price

Judge PRICE IMPACT, not general tone. A headline can be neutral-sounding but
negative for price (e.g. an insider or major holder reducing a stake), or
factual but positive (e.g. a contract win, earnings beat, dividend increase,
regulatory approval). Headlines about OTHER companies, or generic market
commentary, should score near 0.0 unless they clearly affect THIS stock.
"Unusual options activity", "whale trades", short-interest mentions, and
"$X invested N years ago" pieces carry essentially no directional signal:
score them near 0.0.

Score each headline INDEPENDENTLY. Do NOT predict the stock's direction. Do
NOT output any buy/sell/up/down judgment or any aggregate. Return ONLY the
per-headline scores in the required JSON shape, one object per headline, in
the same order the headlines were given.
"""


def build_sentiment_input(row):
    """Sentiment input, headlines only.

    Sentiment_Score/ Sentiment_Trend are left out on purpose so the LLM scores
    each headline on its own rather than anchoring on the window aggregate.

    TODO: the header says "past 5 days" but the window in
    build_sentiment_features.py is news_lookback_days (7 calendar days by
    default). Fixing it changes every prompt, so it needs the teacher data
    regenerated and the eval baseline redone.
    """
    return f"""ASSET: {row['Ticker']}
DATE: {row['Date'].date()}

[NEWS HEADLINES — past 5 days]
{row.get('Top_Headlines', 'No significant headlines.')}"""


def aggregate_sentiment(scores):
    """Turn a list of per-headline scores into (label, view, agg).

    Two zero cases are treated differently:
    - no scorable headlines: no signal, so return NEUTRAL (abstain). Labelling
      these BUY used to add loads of BUYs with no information behind them.
      NEUTRAL is only used at the expert level, PloutosGPT's output and
      Target_Label stay UP/DOWN.
    - headlines that average out to exactly 0.0: that's still a reading, so
      it goes to BUY, same as the paper's Y_d = 1 if p_d >= p_{d-1}.
    """
    clean = []
    for s in scores:
        try:
            v = float(s)
        except (TypeError, ValueError):
            continue
        # clamp, the schema asks for [-1, 1] but the API doesn't always stick to it
        clean.append(max(-1.0, min(1.0, v)))

    n = len(clean)

    if n == 0:
        # abstain
        view = ("NEUTRAL: no scorable headlines; the sentiment expert abstains "
                "(no directional signal for this row).")
        return "NEUTRAL", view, 0.0

    agg = statistics.fmean(clean)
    label = "BUY" if agg >= 0.0 else "SELL"  # non-empty exact-zero -> BUY
    view = (f"{label}: aggregate sentiment {agg:+.3f} "
            f"(mean of {n} headline impact score{'s' if n != 1 else ''}).")
    return label, view, round(agg, 4)


class SentimentExpert(LLMExpert):
    id = "sn"
    name = "Sentiment"
    source_tag = "SENTIMENT"
    prompt_title = "Sentiment Expert"
    signal_column = "Sent_Signal"
    input_column = "Sent_Input"
    view_column = "Sent_View"
    include_basis = False
    data_scope = "single_ticker"
    can_abstain = True            # NEUTRAL when there are no scorable headlines
    teacher_description = "news-headline sentiment"
    persona = SENTIMENT_PERSONA
    schema = SENTIMENT_SCHEMA
    aggregates = True   # LLM scores, signal is worked out in parse_response
    # extra column the generator writes alongside the signal, for analysis
    agg_column = "Sent_Agg_Score"

    def build_features(self, panel: pd.DataFrame, config) -> pd.DataFrame:
        cols = [c for c in REQUIRED_COLUMNS if c in panel.columns]
        return panel[cols].copy()

    def make_input(self, row) -> str:
        return build_sentiment_input(row)

    def parse_response(self, result: dict):
        """result looks like {"scores": [{"headline","score"}, ...]}.
        Raises ResponseSchemaError if 'scores' isn't a list."""
        items = result.get("scores")
        if not isinstance(items, list):
            raise ResponseSchemaError("sentiment response missing 'scores' list")
        raw_scores = [it.get("score") for it in items if isinstance(it, dict)]
        return aggregate_sentiment(raw_scores)


register(SentimentExpert())
