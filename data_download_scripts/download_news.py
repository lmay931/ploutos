import pandas as pd
import requests
import time
import os
import sys
from datetime import timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.secrets import ALPACA_API_KEY, ALPACA_SECRET_KEY

LABELED_DATA_PATH = "data/labeled_training_data.csv"
RAW_NEWS_OUTPUT = "data/raw_headlines.csv"

def fetch_bounded_alpaca_news(ticker, start_str, end_str):
    """Page through the Alpaca news API between start and end dates."""
    all_articles = []
    page_token = None
    
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'accept': 'application/json'
    }
    
    while True:
        url = f"https://data.alpaca.markets/v1beta1/news?sort=desc&symbols={ticker}&limit=50&start={start_str}&end={end_str}"
        if page_token:
            url += f"&page_token={page_token}"
            
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 429:
                print(f"\nRate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
                
            if response.status_code != 200:
                print(f"\nAPI Error for {ticker}: {response.text}")
                break
                
            data = response.json()
            news_items = data.get('news', [])
            
            if not news_items:
                break 
                
            for article in news_items:
                all_articles.append({
                    "Ticker": ticker,
                    "Published": pd.to_datetime(article['created_at']).tz_localize(None), 
                    "Headline": article['headline']
                })
            
            page_token = data.get('next_page_token')
            if not page_token:
                break 
                
            time.sleep(0.35) # rate limit
            
        except Exception as e:
            print(f"\nNetwork error fetching {ticker}: {e}")
            time.sleep(5)
            
    return all_articles

def download_news_pipeline():
    print("[1/2] Extracting boundaries from labeled data...")
    df = pd.read_csv(LABELED_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    
    min_date = df['Date'].min() - timedelta(days=7)
    max_date = df['Date'].max() + timedelta(days=1)
    
    start_str = min_date.strftime('%Y-%m-%dT00:00:00Z')
    end_str = max_date.strftime('%Y-%m-%dT23:59:59Z')
    
    unique_tickers = df['Ticker'].unique()
    print(f"Dataset bounds: {start_str} to {end_str}")
    print(f"Found {len(unique_tickers)} unique tickers. Commencing targeted Alpaca download...")
    
    master_news_list = []
    
    for ticker in tqdm(unique_tickers, desc="Downloading News History"):
        articles = fetch_bounded_alpaca_news(ticker, start_str, end_str)
        master_news_list.extend(articles)
        
    news_df = pd.DataFrame(master_news_list)
    
    if len(news_df) == 0:
        print("No news downloaded. Exiting.")
        return

    # drop duplicates fetched across page boundaries
    news_df = news_df.drop_duplicates(subset=['Ticker', 'Headline', 'Published'])
    
    print(f"\n[2/2] Saving {len(news_df)} historical articles to {RAW_NEWS_OUTPUT}...")
    news_df.to_csv(RAW_NEWS_OUTPUT, index=False)
    print("Download Complete!")

if __name__ == "__main__":
    download_news_pipeline()