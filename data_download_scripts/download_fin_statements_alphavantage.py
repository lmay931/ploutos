import pandas as pd
import requests
import time
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.secrets import ALPHA_VANTAGE_API_KEY

# config
# paths relative to the repo root so it works on any machine
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR  = os.path.join(_REPO_ROOT, 'data')
RAW_DIR   = os.path.join(BASE_DIR, 'raw_alpha_vantage')
FINAL_OUT = os.path.join(BASE_DIR, 'master_quarterly_fundamentals.parquet')

os.makedirs(RAW_DIR, exist_ok=True)

# the three statement endpoints to pull
FUNCTIONS = ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']

def fetch_ticker_data(ticker):
    """Download all 3 statements for a ticker and merge them."""
    ticker_data = {}
    
    for func in FUNCTIONS:
        
        url = f'https://www.alphavantage.co/query?function={func}&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        
        # alpha vantage rate limit check
        if "Information" in data and "rate limit" in data["Information"]:
            print(f"  !! Rate limit hit on {ticker}. Stopping for now.")
            return "LIMIT"
        
        if "quarterlyReports" not in data:
            print(f"  ?? No quarterly data for {ticker} in {func}")
            continue
            
        df = pd.DataFrame(data['quarterlyReports'])
        df['Symbol'] = ticker
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        
        ticker_data[func] = df
        # free tier is 5 requests/ min, so wait 12s between requests
        time.sleep(12) 

    if not ticker_data:
        return None

    # merge the three statements on date and symbol
    try:
        # start from the income statement
        master_ticker_df = ticker_data.get('INCOME_STATEMENT')
        
        if 'BALANCE_SHEET' in ticker_data:
            master_ticker_df = pd.merge(master_ticker_df, ticker_data['BALANCE_SHEET'], 
                                        on=['fiscalDateEnding', 'Symbol'], how='outer', suffixes=('', '_bs'))
        
        if 'CASH_FLOW' in ticker_data:
            master_ticker_df = pd.merge(master_ticker_df, ticker_data['CASH_FLOW'], 
                                        on=['fiscalDateEnding', 'Symbol'], how='outer', suffixes=('', '_cf'))
            
        return master_ticker_df
    except Exception as e:
        print(f"  Error merging {ticker}: {e}")
        return None

# main
# 1. tickers from the price file
df_prices = pd.read_csv(os.path.join(BASE_DIR, 'raw_prices.csv'))
all_tickers = df_prices['Symbol'].unique().tolist()

# 2. skip tickers already downloaded so it can resume
existing_files = [f.replace('.csv', '') for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
tickers_to_pull = [t for t in all_tickers if t not in existing_files]

print(f"Total Tickers: {len(all_tickers)}")
print(f"Already have : {len(existing_files)}")
print(f"Need to pull : {len(tickers_to_pull)}")

newly_downloaded = []

for i, ticker in enumerate(tickers_to_pull):
    # free tier daily limit - 8 tickers (24 requests) is safest but it allows more,
    # 20 tickers is about an hour of runtime
    if i >= 20: 
        print("Reached daily safe limit for this session. Run again tomorrow.")
        break

    print(f"[{i+1}/20] Processing {ticker}...")
    
    result = fetch_ticker_data(ticker)
    
    # "LIMIT" string means we hit the rate limit
    if isinstance(result, str) and result == "LIMIT":
        print("Stopping due to API rate limit.")
        break
    
    if result is not None:
        # save each ticker as a checkpoint
        ticker_out = os.path.join(RAW_DIR, f"{ticker}.csv")
        result.to_csv(ticker_out, index=False)
        newly_downloaded.append(result)
        print(f"  Saved {ticker}")
    else:
        # empty file so failed tickers aren't retried
        open(os.path.join(RAW_DIR, f"{ticker}.csv"), 'a').close()
        print(f"  No data for {ticker}, skipping in future.")

# 3. compile all raw files into the master parquet
print("\n[INFO] Compiling all raw files into Master Parquet...")
all_dfs = []
for f in os.listdir(RAW_DIR):
    if f.endswith('.csv') and os.path.getsize(os.path.join(RAW_DIR, f)) > 0:
        all_dfs.append(pd.read_csv(os.path.join(RAW_DIR, f)))

if all_dfs:
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_parquet(FINAL_OUT, engine='pyarrow')
    print(f"SUCCESS: {FINAL_OUT} updated. Total records: {len(master_df)}")
else:
    print("No data to compile.")