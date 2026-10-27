"""
Joins the fundamental, sentiment and macro features onto the labelled price
data, adds the technical features (MA20, relative momentum), recomputes the
price-dependent fundamental ratios against the current Close, and writes the
master CSV that everything downstream reads.

Fundamental ratios:
   build_fundamental_features.py gives TTM eps_ttm and fcf_ttm, the ratios that
   don't depend on price (Gross_Margin, ROE, ROA, Current_Ratio) and two flags
   (Is_Financial, ROE_distorted). P/E, Market_Cap and FCF_Yield are computed
   here from the forward-filled TTM inputs and each day's Close.

   - TTM rather than a quarter x4: the fundamental expert is mostly there to
     give valuation context for the price/ sentiment signals over 5 days, so
     a stable anchor matters more than the latest quarter. Annualising a
     single quarter makes P/E swing a lot on one-off items.
   - recomputing here: a forward-filled P/E would stay frozen at the filing
     date value even as Close moves. computing it daily fixes that.
   - eps_ttm <= 0 (loss over the trailing year): P/E is NaN, shown to the LLM
     as "N/A (loss-making period)" so it isn't read as missing data.
   - buyback-distorted equity (e.g. HD): ROE is nulled upstream with
     ROE_distorted=True and shown as "N/A (buyback-distorted equity)". ROA is
     still there for these.
   - financials/ REITs: Gross_Margin, Current_Ratio and fcf_ttm are nulled
     upstream (Is_Financial=True), so FCF_Yield is NaN as well. P/E, ROE and
     ROA are kept. shown as "N/A (not meaningful for financials)".

Train/ test split:
  - this script only outputs TRAIN_START..TRAIN_END
  - 2026-01-01 onwards is the held out test set, built separately
  - the window covers a few different regimes (2022 bear, 2023 recovery, 2024
    late-cycle rally, 2025 Iran shock)
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig

_cfg = PipelineConfig()

# =====================================================================
# date window, train + validation only
# =====================================================================
TRAIN_START = pd.Timestamp(_cfg.data_start)
TRAIN_END   = pd.Timestamp(_cfg.data_end)

# =====================================================================
# warn if UP/DOWN skew is above this
# =====================================================================
CLASS_SKEW_WARN_THRESHOLD = 0.10

# =====================================================================
# sanity ranges for the price-dependent ratios
# =====================================================================
PE_VALID_RANGE = (0.5, 500)
FCF_YIELD_VALID_RANGE = (-0.5, 0.5)

# =====================================================================
# newer fundamental columns. if an existing master CSV doesn't have these
# it gets fully rebuilt, otherwise appending would misalign the columns
# =====================================================================
NEW_FUNDAMENTAL_COLS = ['ROA', 'Is_Financial', 'ROE_distorted']

# =====================================================================
# output
# =====================================================================
OUTPUT_PATH = "data/master_training_data.csv"


def build_master_training_dataset():
    print("[INFO] Building Master Feature Store...")
    print(f"   Window: {TRAIN_START.date()} .. {TRAIN_END.date()}")

    # 0. check for an existing file and which tickers are already done
    existing_tickers = set()
    file_exists = os.path.exists(OUTPUT_PATH)

    if file_exists:
        print(f"[INFO] Found existing master file at {OUTPUT_PATH}. Checking schema + tickers...")
        try:
            # check the schema first, if the new fundamental columns are
            # missing appending would misalign things so rebuild instead
            existing_header = pd.read_csv(OUTPUT_PATH, nrows=0).columns.tolist()
            missing_new = [c for c in NEW_FUNDAMENTAL_COLS if c not in existing_header]
            if missing_new:
                print(f"[WARNING] Existing master CSV lacks new columns "
                      f"{missing_new}. Forcing full rebuild to keep schema "
                      f"consistent. If you want to preserve the existing rows, "
                      f"move/rename the file first and re-merge manually.")
                file_exists = False
            else:
                existing_df = pd.read_csv(OUTPUT_PATH, usecols=['Ticker'])
                existing_tickers = set(existing_df['Ticker'].unique())
                print(f"   -> Schema OK. {len(existing_tickers):,} tickers already processed.")
        except Exception as e:
            print(f"[WARNING] Could not read existing master CSV. Proceeding "
                  f"with full rebuild. Error: {e}")
            file_exists = False

    # 1. load data
    print("Loading raw datasets...")
    prices_df = pd.read_csv("data/labeled_training_data.csv")

    # 1b. only keep new tickers before doing anything expensive
    all_input_tickers = set(prices_df['Ticker'].unique())
    new_tickers = all_input_tickers - existing_tickers

    if file_exists:
        if not new_tickers:
            print("[INFO] No new tickers to process. Dataset is up to date. Exiting.")
            return

        print(f"[INFO] Processing {len(new_tickers):,} NEW tickers...")
        prices_df = prices_df[prices_df['Ticker'].isin(new_tickers)]

    fund_df = pd.read_parquet("data/fundamental_expert_features.parquet")
    sent_df = pd.read_parquet("data/sentiment_expert_features.parquet")
    macro_df = pd.read_parquet("data/macro_expert_features.parquet")

    # filter the expert dataframes too to save memory in the merge
    if file_exists:
        if 'Ticker' in fund_df.columns:
            fund_df = fund_df[fund_df['Ticker'].isin(new_tickers)]
        if 'Ticker' in sent_df.columns:
            sent_df = sent_df[sent_df['Ticker'].isin(new_tickers)]

    # 2. make sure dates are datetimes
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    fund_df['Date'] = pd.to_datetime(fund_df['Date'])
    sent_df['Date'] = pd.to_datetime(sent_df['Date'])
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])

    # 3. merge
    print("Merging datasets cleanly...")

    # fundamental: TTM inputs, non-price ratios and flags. P/E, FCF_Yield and
    # Market_Cap get computed below
    fund_cols = [
        'Ticker', 'Date', 'Filing_Date','fiscalDateEnding',
        # TTM inputs for downstream P/E and FCF_Yield computation
        'eps_ttm', 'fcf_ttm', 'commonStockSharesOutstanding',
        # Pre-computed non-price ratios
        'Gross_Margin', 'ROE', 'ROA', 'Current_Ratio',
        # Flags for downstream rendering / filtering
        'Is_Financial', 'ROE_distorted',
    ]
    fund_cols_present = [c for c in fund_cols if c in fund_df.columns]
    missing_fund = set(fund_cols) - set(fund_cols_present)
    if missing_fund:
        print(f"[WARNING] fund_df is missing expected columns: {missing_fund}. "
              f"Has fundamental_features.py been rerun against the updated "
              f"version? Continuing without them.")
    master_df = pd.merge(prices_df, fund_df[fund_cols_present],
                         on=['Ticker', 'Date'], how='left')

    sent_cols = ['Ticker', 'Date', 'Sentiment_Score', 'Sentiment_Trend', 'Top_Headlines']
    sent_cols_present = [c for c in sent_cols if c in sent_df.columns]
    master_df = pd.merge(master_df, sent_df[sent_cols_present],
                         on=['Ticker', 'Date'], how='left')

    macro_cols = ['Date', 'SPY_5d_Return', 'SPY_Vol_20d']
    macro_cols_present = [c for c in macro_cols if c in macro_df.columns]
    master_df = pd.merge(master_df, macro_df[macro_cols_present],
                         on='Date', how='left')

    # 4. forward fill within each ticker
    print("Forward-filling expert features...")
    master_df = master_df.sort_values(by=['Ticker', 'Date'])

    stock_ffill_cols = [
        # TTM fundamental inputs (drive P/E + FCF_Yield recomputation)
        'eps_ttm', 'fcf_ttm', 'commonStockSharesOutstanding',
        # Pre-computed non-price ratios
        'Gross_Margin', 'ROA', 'Current_Ratio',
        # Metadata about the source filing
        'Filing_Date', 'fiscalDateEnding',
        # buyback distortion flag (per filing, ffilled with ROE)
        'ROE_distorted',
        # Sentiment
        'Sentiment_Score', 'Sentiment_Trend',
    ]
    for col in stock_ffill_cols:
        if col in master_df.columns:
            master_df[col] = master_df.groupby('Ticker')[col].ffill()

    # Is_Financial is constant per ticker, so ffill + bfill fills every row
    # including ones the merge didn't match. tickers not in fund_df at all
    # default to False
    if 'Is_Financial' in master_df.columns:
        master_df['Is_Financial'] = (
            master_df.groupby('Ticker')['Is_Financial']
            .transform(lambda s: s.ffill().bfill())
        )
        master_df['Is_Financial'] = (
            master_df['Is_Financial'].fillna(False).astype(bool)
        )
    else:
        master_df['Is_Financial'] = False

    # ROE_distorted is per filing, missing = not distorted
    if 'ROE_distorted' in master_df.columns:
        master_df['ROE_distorted'] = (
            master_df['ROE_distorted'].fillna(False).astype(bool)
        )
    else:
        master_df['ROE_distorted'] = False

    # don't forward fill headlines
    master_df['Top_Headlines'] = master_df['Top_Headlines'].fillna(
        'No significant headlines.'
    )

    # 5. recompute the price-dependent ratios against the current Close
    print("Recomputing price-dependent fundamental ratios from TTM inputs...")

    master_df['Market_Cap'] = (
        master_df['Close'] * master_df['commonStockSharesOutstanding']
    )

    master_df['PE_Ratio'] = np.where(
        master_df['eps_ttm'] > 0,
        master_df['Close'] / master_df['eps_ttm'],
        np.nan,
    )

    # NaN for financials since fcf_ttm is blanked upstream
    master_df['FCF_Yield'] = np.where(
        master_df['Market_Cap'] > 0,
        master_df['fcf_ttm'] / master_df['Market_Cap'],
        np.nan,
    )

    bad_pe = master_df['PE_Ratio'].notna() & ~master_df['PE_Ratio'].between(*PE_VALID_RANGE)
    bad_fcf = master_df['FCF_Yield'].notna() & ~master_df['FCF_Yield'].between(*FCF_YIELD_VALID_RANGE)
    n_bad_pe = bad_pe.sum()
    n_bad_fcf = bad_fcf.sum()
    if n_bad_pe > 0:
        master_df.loc[bad_pe, 'PE_Ratio'] = np.nan
    if n_bad_fcf > 0:
        master_df.loc[bad_fcf, 'FCF_Yield'] = np.nan
    if n_bad_pe + n_bad_fcf > 0:
        print(f"   Masked {n_bad_pe:,} out-of-range P/E and "
              f"{n_bad_fcf:,} out-of-range FCF_Yield to NaN.")

    for col in ['SPY_5d_Return', 'SPY_Vol_20d']:
        if col in master_df.columns:
            master_df[col] = master_df[col].ffill()

    # 6. technical features
    print("Calculating technical overlays (MA20, Relative Momentum)...")

    master_df['MA20'] = master_df.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(20).mean()
    )
    master_df['Price_vs_MA20'] = (master_df['Close'] - master_df['MA20']) / master_df['MA20']
    master_df['Stock_5d_Return'] = master_df.groupby('Ticker')['Close'].pct_change(5)

    if 'SPY_5d_Return' in master_df.columns:
        master_df['Relative_Momentum'] = (
            master_df['Stock_5d_Return'] - master_df['SPY_5d_Return']
        )

    # 7. drop rows missing important features.
    # every row needs Price_vs_MA20, ROA, Sentiment_Score, SPY_5d_Return, and
    # non-financials also need Gross_Margin and Current_Ratio.
    # ROE isn't required because it's nulled for buyback-distorted names like
    # HD, and I still want those in the training set
    initial_len = len(master_df)

    universal_required = ['Price_vs_MA20', 'ROA', 'Sentiment_Score', 'SPY_5d_Return']
    non_financial_required = ['Gross_Margin', 'Current_Ratio']

    universal_required = ['Price_vs_MA20', 'ROA', 'Sentiment_Score', 'SPY_5d_Return']
    non_fin_required = ['Gross_Margin', 'Current_Ratio']
    is_nonfin = ~master_df['Is_Financial'].fillna(False).astype(bool)
    for col in universal_required:
        n = master_df[col].isna().sum()
        print(f"   {col}: {n:,} NaN ({n/len(master_df)*100:.1f}%)")
    for col in non_fin_required:
        n = master_df.loc[is_nonfin, col].isna().sum()
        print(f"   {col} (non-fin only): {n:,} NaN ({n/is_nonfin.sum()*100:.1f}%)")


    keep_mask = master_df[universal_required].notna().all(axis=1)
    keep_mask &= (
        master_df['Is_Financial']
        | master_df[non_financial_required].notna().all(axis=1)
    )
    master_df = master_df[keep_mask].copy()

    n_dropped = initial_len - len(master_df)
    print(f"Dropped {n_dropped:,} rows missing critical features "
          f"(universal + non-financial industrial template).")

    # 8. date window
    pre_window_len = len(master_df)
    master_df = master_df[
        (master_df['Date'] >= TRAIN_START) & (master_df['Date'] <= TRAIN_END)
    ].copy()
    print(f"Date window filter: kept {len(master_df):,} rows from "
          f"{TRAIN_START.date()}..{TRAIN_END.date()} "
          f"(dropped {pre_window_len - len(master_df):,} outside window).")

    # nothing left to save
    if master_df.empty:
        print("\n[INFO] No valid rows remained after filtering. Nothing to save.")
        return

    # 9. diagnostics (only for the tickers processed in this run)
    print("\n--- Diagnostics (Current Run) ---")

    yearly = master_df.assign(Year=master_df['Date'].dt.year).groupby('Year').size()
    print("Rows per year:")
    for year, n in yearly.items():
        print(f"   {year}: {n:,}")

    if 'Target_Label' in master_df.columns:
        label_counts = master_df['Target_Label'].value_counts()
        total_directional = label_counts.get('UP', 0) + label_counts.get('DOWN', 0)
        print(f"\nLabel distribution:")
        for label in ['UP', 'DOWN', 'FLAT']:
            n = label_counts.get(label, 0)
            pct = (n / len(master_df) * 100) if len(master_df) else 0
            print(f"   {label:5s}: {n:,} ({pct:.1f}%)")

        if total_directional > 0:
            up = label_counts.get('UP', 0)
            down = label_counts.get('DOWN', 0)
            skew = abs(up - down) / total_directional
            if skew > CLASS_SKEW_WARN_THRESHOLD:
                print(f"\n   WARN: UP/DOWN skew is {skew*100:.1f}% "
                      f"(>{CLASS_SKEW_WARN_THRESHOLD*100:.0f}%). "
                      f"Consider undersampling at teacher_model.")

    print(f"\nTicker coverage: {master_df['Ticker'].nunique()} tickers")
    rows_per_ticker = master_df.groupby('Ticker').size()
    print(f"   min rows/ticker: {rows_per_ticker.min()}")
    print(f"   max rows/ticker: {rows_per_ticker.max()}")
    print(f"   median:          {int(rows_per_ticker.median())}")

    # fundamental coverage
    pe_valid = master_df['PE_Ratio'].notna().sum()
    pe_na = master_df['PE_Ratio'].isna().sum()
    print(f"\nP/E coverage: {pe_valid:,} valid / "
          f"{pe_na:,} N/A (loss-making period or filtered)")

    if 'ROA' in master_df.columns:
        roa_valid = master_df['ROA'].notna().sum()
        roa_na = master_df['ROA'].isna().sum()
        print(f"ROA coverage: {roa_valid:,} valid / {roa_na:,} N/A")

    if 'ROE_distorted' in master_df.columns:
        n_distorted_rows = int(master_df['ROE_distorted'].sum())
        n_distorted_tickers = (
            master_df.loc[master_df['ROE_distorted'], 'Ticker'].nunique()
        )
        print(f"ROE distorted (buyback-compressed equity): "
              f"{n_distorted_rows:,} rows across {n_distorted_tickers} tickers")

    if 'Is_Financial' in master_df.columns:
        n_fin_rows = int(master_df['Is_Financial'].sum())
        n_fin_tickers = (
            master_df.loc[master_df['Is_Financial'], 'Ticker'].nunique()
        )
        print(f"Financial/REIT rows: {n_fin_rows:,} "
              f"across {n_fin_tickers} tickers (GM/Current_Ratio/FCF suppressed)")

    # 10. save/ append
    print(f"\n{'=' * 60}")
    if file_exists:
        print(f"Appending {len(master_df):,} rows to existing master dataset...")
        master_df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
    else:
        print(f"Creating new master dataset...")
        master_df.to_csv(OUTPUT_PATH, index=False)

    print(f"   Tickers processed: {master_df['Ticker'].nunique()}")
    print(f"   Output           : {OUTPUT_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_master_training_dataset()