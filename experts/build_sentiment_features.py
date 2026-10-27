import os
import sys
import pandas as pd
from datetime import timedelta
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

LABELED_DATA_PATH = "data/labeled_training_data.csv"
RAW_NEWS_INPUT = "data/raw_headlines.csv"
OUTPUT_PARQUET_PATH = "data/sentiment_expert_features.parquet"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig

def select_top_headlines(window_headlines, max_headlines=5):
    """
    Pick the top headlines for the prompt, by how extreme their FinBERT score
    is, preferring headlines from different days.
    """
    if window_headlines.empty:
        return "No significant headlines in this window."

    window_headlines = window_headlines.copy()
    window_headlines['Extremity'] = window_headlines['Headline_Score'].abs()
    window_headlines = window_headlines.sort_values('Extremity', ascending=False)

    selected = []
    dates_covered = set()

    for _, row in window_headlines.iterrows():
        if len(selected) >= max_headlines:
            break

        day = row['Published'].date()
        # one headline per day where possible, but the first 3 can share a day
        if day not in dates_covered or len(selected) < 3:
            # just the text, scores aren't included in the prompt
            selected.append(f"  {day.strftime('%Y-%m-%d')}: {row['Headline']}")
            dates_covered.add(day)

    return "\n".join(selected)


def build_sentiment_features(news_lookback_days=7,
                              highly_threshold=0.35,
                              cautiously_threshold=0.10):
    print("[1/4] Loading Data...")
    df_labels = pd.read_csv(LABELED_DATA_PATH)
    df_labels['Date'] = pd.to_datetime(df_labels['Date'])

    df_news = pd.read_csv(RAW_NEWS_INPUT)
    df_news['Published'] = pd.to_datetime(df_news['Published'])

    df_news['Headline'] = df_news['Headline'].astype(str).fillna('')
    df_news = df_news[df_news['Headline'].str.lower() != 'nan'].copy()

    print("\n[2/4] Loading FinBERT & Scoring All Raw Headlines...")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        truncation=True,     
        max_length=512
    )

    headlines = df_news['Headline'].tolist()
    headlines_truncated = [h[:512] for h in headlines]

    scores = []

    label_map = {'positive': 1, 'negative': -1, 'neutral': 0}

    for out in tqdm(
        finbert(headlines_truncated, batch_size=32),
        total=len(headlines_truncated),
        desc="FinBERT Scoring"
    ):
        direction = label_map.get(out['label'].lower(), 0)
        scores.append(direction * out['score'])

    df_news['Headline_Score'] = scores

    print("\n[3/4] Calculating 5-Day Rolling Sentiment & Extracting Top Headlines...")
    sentiment_records = []
    news_by_ticker = dict(tuple(df_news.groupby('Ticker')))

    for index, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Processing Dates"):
        ticker = row['Ticker']
        target_date = row['Date']

        start_date = target_date - timedelta(days=news_lookback_days)

        net_score = 0.0
        sentiment_trend = "Neutral / Mixed"
        top_headlines_str = "No significant headlines."

        if ticker in news_by_ticker:
            t_news = news_by_ticker[ticker]
            mask = (t_news['Published'] >= start_date) & (t_news['Published'] <= target_date)
            window_news = t_news.loc[mask].copy()

            if not window_news.empty:
                # mean score over the whole window
                net_score = window_news['Headline_Score'].mean()

                # top headlines for the prompt (text only)
                top_headlines_str = select_top_headlines(window_news)

                # trend label from the mean score
                if net_score >= highly_threshold:
                    sentiment_trend = "Highly Bullish"
                elif net_score <= -highly_threshold:
                    sentiment_trend = "Highly Bearish"
                elif net_score > cautiously_threshold:
                    sentiment_trend = "Cautiously Optimistic"
                elif net_score < -cautiously_threshold:
                    sentiment_trend = "Cautiously Pessimistic"
                # else remains "Neutral / Mixed"

        sentiment_records.append({
            "Date": target_date,
            "Ticker": ticker,
            "Sentiment_Score": round(net_score, 4),
            "Sentiment_Trend": sentiment_trend,
            "Top_Headlines": top_headlines_str
        })

    print("\n[4/4] Saving Expert Features to Parquet...")
    final_df = pd.DataFrame(sentiment_records)
    final_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    print(f"\n{'=' * 50}")
    print(f"Sentiment features saved.")
    print(f"   Rows processed : {len(final_df)}")
    print(f"   Tickers covered: {final_df['Ticker'].nunique()}")
    print(f"   Date range     : {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
    print(f"   Output         : {OUTPUT_PARQUET_PATH}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    cfg = PipelineConfig()
    build_sentiment_features(
        news_lookback_days=cfg.news_lookback_days,
        highly_threshold=cfg.sentiment_highly_threshold,
        cautiously_threshold=cfg.sentiment_cautiously_threshold,
    )