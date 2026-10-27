import pandas as pd
import requests
import time
import os
import sys
from datetime import datetime, timedelta
from tqdm import tqdm

# so core.secrets can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.secrets import ALPACA_API_KEY, ALPACA_SECRET_KEY

OUTPUT_CSV = "data/spy_prices.csv"

def fetch_alpaca_bars(symbol, start_dt, end_dt):
    """
    Page through the Alpaca v2 bars API to get the full history.
    """
    all_bars = []
    next_page_token = None
    
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY,
        'accept': 'application/json'
    }
    
    # alpaca wants RFC-3339 dates
    start_str = start_dt.strftime('%Y-%m-%dT00:00:00Z')
    end_str = end_dt.strftime('%Y-%m-%dT23:59:59Z')
    
    print(f"[INFO] Fetching {symbol} bars from {start_str} to {end_str}")

    while True:
        url = (
            f"https://data.alpaca.markets/v2/stocks/bars"
            f"?symbols={symbol}&timeframe=1D&start={start_str}&end={end_str}"
            f"&limit=1000&adjustment=all&feed=sip&sort=asc"
        )
        
        if next_page_token:
            url += f"&page_token={next_page_token}"
            
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 429:
                print("\nRate limit hit. Sleeping 60s...")
                time.sleep(60)
                continue
                
            if response.status_code != 200:
                print(f"\nAPI Error: {response.text}")
                break
                
            data = response.json()
            bars = data.get('bars', {}).get(symbol, [])
            
            if not bars:
                break
                
            for b in bars:
                all_bars.append({
                    "Date": pd.to_datetime(b['t']).tz_localize(None).date(),
                    "Open": b['o'],
                    "High": b['h'],
                    "Low": b['l'],
                    "Close": b['c'],
                    "Volume": b['v']
                })
            
            next_page_token = data.get('next_page_token')
            if not next_page_token:
                break
                
            time.sleep(0.3) # rate limit
            
        except Exception as e:
            print(f"\nNetwork error: {e}")
            time.sleep(5)
            
    return all_bars

def main():
    # 10 year lookback
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * 10)
    
    spy_data = fetch_alpaca_bars("SPY", start_date, end_date)
    
    if spy_data:
        df = pd.DataFrame(spy_data)
        # in case pages overlap
        df = df.drop_duplicates(subset=['Date'])
        df = df.sort_values('Date')
        
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nSUCCESS: Downloaded {len(df)} days of SPY data.")
        print(f"Saved to: {OUTPUT_CSV}")
    else:
        print("\nFailed to retrieve SPY data.")

if __name__ == "__main__":
    main()