"""
Builds the fundamental expert features by joining the AlphaVantage quarterly
statements (already merged into one parquet) onto daily prices.

Some notes on how it works:

1) The parquet at FUNDAMENTALS_PARQUET has the income statement, balance sheet
   and cash flow joined on (Symbol, fiscalDateEnding), so that's the only input.

2) AlphaVantage only gives fiscalDateEnding (end of the period), not the date
   the results were actually released. Merging on that would let the model see
   Q1 numbers on Apr 1 when they aren't public until late Apr/ early May, so I
   add EARNINGS_LAG_DAYS (45) to get an estimated Filing_Date and merge on that.

3) Earnings and cash flow are TTM rather than a single quarter x4. Annualising
   one quarter makes P/E jump around whenever there's a one-off item (e.g. the
   AT&T Q3 2025 DirecTV gain took P/E from 11 to 3 for a quarter). TTM gives a
   much more stable valuation anchor, which is mainly what the fundamental
   expert is for.

4) Price-dependent ratios (P/E, FCF yield, market cap) are computed later in
   build_master_training_dataset.py after forward filling the inputs. If you
   ffill a precomputed P/E it stays frozen even though Close keeps moving.

5) ROA as well as ROE. ROE = net income/ book equity blows up when buybacks
   shrink book equity towards zero (e.g. Home Depot, ROE ~1500%). ROA uses
   total assets which buybacks don't really affect. Distorted ROE values are
   nulled and flagged rather than dropping the row, since the rest of the
   company's data is still useful.

6) For banks, insurers, capital markets firms and REITs, gross margin (no COGS
   so ~100%), current ratio and FCF yield don't really mean anything, so they
   get blanked rather than shown to the LLM. P/E, ROE and ROA are kept and the
   tickers stay in the universe.

Foreign ADRs are dropped: if a ticker ever reports in a non-USD currency it's
removed, since the ratios aren't reliable without a proper FX adjustment.

NaN handling:
   - rows with NaN fiscalDateEnding are dropped on load
   - rows before the first filing get bfilled from the next filing (up to ~6
     weeks of leakage, but the date window downstream drops most of these)
   - rows with no fundamental coverage at all are dropped
   - eps_ttm and fcf_ttm are NaN for the first 3 quarters of each ticker (not
     enough history), which is fine as those are outside the training window
   - the suppressed financial metrics are NaN on purpose, and distorted ROE
     is nulled with ROE_distorted = True
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig

_cfg = PipelineConfig()

# --- config ---
DATA_DIR = 'data'
PRICE_FILE = 'raw_prices.csv'
FUNDAMENTALS_PARQUET = 'master_quarterly_fundamentals.parquet'
OUTPUT_FILE = 'data/fundamental_expert_features.parquet'

EARNINGS_LAG_DAYS    = _cfg.earnings_lag_days
TTM_WINDOW           = _cfg.ttm_window
ROE_PLAUSIBILITY_CAP = _cfg.roe_plausibility_cap

# banks, insurers, capital markets firms and REITs (see note 6 above). only
# gross margin, current ratio and FCF yield are blanked, they stay in the universe.
# this is a hand-written list because the parquet doesn't have a sector column,
# so it needs updating if the universe changes. if a 'Sector' column is
# available _resolve_financial_tickers uses that as well
FINANCIAL_TICKERS = {
    # Banks
    'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'GS', 'MS',
    'HSBC', 'BCS', 'RY', 'TD', 'COF', 'BK', 'STT',
    # Insurers
    'AIG', 'MET', 'PRU', 'ALL', 'TRV', 'PGR', 'CB', 'AFL', 'HIG',
    # Capital markets / exchanges / asset managers
    'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'NDAQ',
    # REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'PSA', 'WELL', 'DLR', 'VICI',
}

# metrics that get blanked for the tickers above
FINANCIAL_SUPPRESS_COLS = ['Gross_Margin', 'Current_Ratio', 'fcf_ttm']

RAW_COLS = [
    'Symbol',
    'fiscalDateEnding',
    'reportedCurrency',  # needed for the ADR exclusion
    'grossProfit',
    'totalRevenue',
    'netIncome',
    'totalAssets',
    'totalCurrentAssets',
    'totalCurrentLiabilities',
    'totalShareholderEquity',
    'commonStockSharesOutstanding',
    'operatingCashflow',
    'capitalExpenditures',
]

NUMERIC_COLS = [
    'grossProfit', 'totalRevenue', 'netIncome',
    'totalAssets',
    'totalCurrentAssets', 'totalCurrentLiabilities',
    'totalShareholderEquity', 'commonStockSharesOutstanding',
    'operatingCashflow', 'capitalExpenditures',
]


def _resolve_financial_tickers(df_funds):
    """Return the set of financial/ REIT tickers.

    Uses the 'Sector' column if the parquet has one (plus the manual list),
    otherwise just FINANCIAL_TICKERS.
    """
    if 'Sector' in df_funds.columns:
        sector = df_funds['Sector'].astype(str).str.lower()
        is_fin = sector.str.contains('financ') | sector.str.contains('real estate')
        derived = set(df_funds.loc[is_fin, 'Ticker'].unique())
        # union with the manual list so nothing in it gets missed
        return derived | FINANCIAL_TICKERS
    return set(FINANCIAL_TICKERS)


def build_fundamental_features():
    print("[1/5] Loading inputs...")

    df_prices = pd.read_csv(os.path.join(DATA_DIR, PRICE_FILE))
    if 'Symbol' in df_prices.columns:
        df_prices = df_prices.rename(columns={'Symbol': 'Ticker'})
    df_prices['Date'] = pd.to_datetime(df_prices['Date'], format="%Y-%m-%d")
    print(f"   Prices: {len(df_prices):,} rows, "
          f"{df_prices['Ticker'].nunique()} tickers, "
          f"{df_prices['Date'].min().date()} .. {df_prices['Date'].max().date()}")

    df_funds = pd.read_parquet(os.path.join(DATA_DIR, FUNDAMENTALS_PARQUET))

    # keep Sector if it's there, used to find the financials
    raw_cols = list(RAW_COLS)
    if 'Sector' in df_funds.columns and 'Sector' not in raw_cols:
        raw_cols.append('Sector')

    missing = [c for c in raw_cols if c not in df_funds.columns]
    if missing:
        raise KeyError(
            f"Parquet is missing expected AlphaVantage columns: {missing}. "
            f"(Note: 'totalAssets' is now required for ROA and 'reportedCurrency' "
            f"is required for ADR exclusion - add them to your parquet build if absent.)"
        )

    df_funds = df_funds[raw_cols].rename(columns={'Symbol': 'Ticker'}).copy()
    print(f"   Fundamentals: {len(df_funds):,} quarterly rows, "
          f"{df_funds['Ticker'].nunique()} tickers")

    # drop foreign ADRs that report in their local currency
    # clean up the currency column first (NaNs, whitespace, case)
    curr_clean = df_funds['reportedCurrency'].fillna('').astype(str).str.strip().str.upper()
    
    # rows where the currency is filled in but isn't USD (ignoring literal 'NAN'/'NONE' strings)
    non_usd_mask = (curr_clean != '') & (~curr_clean.isin(['NAN', 'NONE'])) & (curr_clean != 'USD')
    
    excluded_tickers = set(df_funds.loc[non_usd_mask, 'Ticker'].unique())
    
    if excluded_tickers:
        df_prices = df_prices[~df_prices['Ticker'].isin(excluded_tickers)]
        df_funds = df_funds[~df_funds['Ticker'].isin(excluded_tickers)]
        print(f"   Excluded {len(excluded_tickers)} foreign ADRs "
              f"(local-currency reporting): {sorted(list(excluded_tickers))}")

    # financials/ REITs, these stay in the universe but get some metrics blanked
    financial_tickers = _resolve_financial_tickers(df_funds)

    before = len(df_funds)
    df_funds = df_funds.dropna(subset=['fiscalDateEnding'])
    n_dropped_nan_dates = before - len(df_funds)
    if n_dropped_nan_dates > 0:
        print(f"   Dropped {n_dropped_nan_dates:,} fundamental rows with "
              f"NaN fiscalDateEnding (cannot determine filing date).")

    print("[2/5] Applying earnings lag, coercing types, computing TTM rollups...")
    df_funds['fiscalDateEnding'] = pd.to_datetime(df_funds['fiscalDateEnding'], format="%Y-%m-%d")

    df_funds['Filing_Date'] = (
        df_funds['fiscalDateEnding'] + pd.Timedelta(days=EARNINGS_LAG_DAYS)
    )

    for c in NUMERIC_COLS:
        df_funds[c] = pd.to_numeric(df_funds[c], errors='coerce')

    # quarterly EPS. AlphaVantage doesn't have a clean diluted EPS for every
    # ticker so I use basic EPS = netIncome/ commonStockSharesOutstanding.
    # dilution is tiny for mega caps anyway. a loss quarter gives negative eps
    df_funds['eps_quarterly'] = np.where(
        df_funds['commonStockSharesOutstanding'] > 0,
        df_funds['netIncome'] / df_funds['commonStockSharesOutstanding'],
        np.nan,
    )

    # quarterly FCF = operating cash flow - |capex| (AlphaVantage reports capex
    # as a positive number)
    df_funds['fcf_quarterly'] = (
        df_funds['operatingCashflow'] - df_funds['capitalExpenditures'].abs()
    )

    # TTM = sum of the last 4 quarters per ticker. min_periods=TTM_WINDOW so
    # nothing is defined until there are 4 quarters of history
    df_funds = df_funds.sort_values(['Ticker', 'fiscalDateEnding'])

    df_funds['eps_ttm'] = (
        df_funds.groupby('Ticker')['eps_quarterly']
        .rolling(window=TTM_WINDOW, min_periods=TTM_WINDOW)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df_funds['fcf_ttm'] = (
        df_funds.groupby('Ticker')['fcf_quarterly']
        .rolling(window=TTM_WINDOW, min_periods=TTM_WINDOW)
        .sum()
        .reset_index(level=0, drop=True)
    )

    n_eps_ttm_valid = df_funds['eps_ttm'].notna().sum()
    print(f"   TTM EPS computed: {n_eps_ttm_valid:,} valid quarterly rows "
          f"(first {TTM_WINDOW - 1} quarters per ticker are NaN by design).")

    # ratios that don't depend on price. these only use filing data so it's
    # fine to compute them here and ffill later
    df_funds['Gross_Margin'] = np.where(
    df_funds['totalRevenue'] != 0,
    df_funds['grossProfit'] / df_funds['totalRevenue'],
    np.nan,
    )

    gm_artifact = df_funds['Gross_Margin'] >= 0.999
    n_gm_artifact = int(gm_artifact.sum())
    if n_gm_artifact:
        df_funds.loc[gm_artifact, 'Gross_Margin'] = np.nan
        print(f"   Nulled {n_gm_artifact:,} gross-margin artifacts (>=99.9%, "
              f"grossProfit==totalRevenue, no separable COGS).")

    # TTM net income, used for both ROA and ROE. keep this outside the
    # gross margin artifact branch above or ROA/ ROE go missing
    df_funds['netIncome_ttm'] = (
        df_funds.groupby('Ticker')['netIncome']
        .rolling(window=TTM_WINDOW, min_periods=TTM_WINDOW)
        .sum()
        .reset_index(level=0, drop=True)
    )
    # ROA = TTM net income/ total assets. not affected by buybacks like ROE is,
    # so this is the main capital efficiency metric
    df_funds['ROA'] = np.where(
        df_funds['totalAssets'] > 0,
        df_funds['netIncome_ttm'] / df_funds['totalAssets'],
        np.nan,
    )

    # ROE = TTM net income/ book equity. only makes sense with positive equity,
    # and even then buybacks can push equity close to zero (HD ~1500%). values
    # above the cap or with non-positive equity get nulled and flagged, but the
    # row is kept since the rest of its data is still fine
    raw_roe = np.where(
        df_funds['totalShareholderEquity'] > 0,
        df_funds['netIncome_ttm'] / df_funds['totalShareholderEquity'],
        np.nan,
    )
    distorted = (
        (df_funds['totalShareholderEquity'] <= 0)
        | (np.abs(raw_roe) > ROE_PLAUSIBILITY_CAP)
    )
    df_funds['ROE'] = np.where(
        np.abs(raw_roe) > ROE_PLAUSIBILITY_CAP, np.nan, raw_roe
    )
    df_funds['ROE_distorted'] = distorted.astype(bool)
    n_roe_distorted = int(df_funds['ROE_distorted'].sum())
    print(f"   ROE distorted (non-positive or buyback-compressed equity): "
          f"{n_roe_distorted:,} rows -> ROE nulled, ROA retained.")

    df_funds['Current_Ratio'] = np.where(
        df_funds['totalCurrentLiabilities'] > 0,
        df_funds['totalCurrentAssets'] / df_funds['totalCurrentLiabilities'],
        np.nan,
    )

    # blank the metrics that don't mean anything for financials/ REITs (e.g. a
    # bank's ~100% gross margin) so the LLM never sees them. P/E, ROE, ROA kept
    fin_mask = df_funds['Ticker'].isin(financial_tickers)
    n_fin_rows = int(fin_mask.sum())
    if n_fin_rows:
        df_funds.loc[fin_mask, FINANCIAL_SUPPRESS_COLS] = np.nan
        print(f"   Suppressed {FINANCIAL_SUPPRESS_COLS} for {n_fin_rows:,} "
              f"financial/REIT rows "
              f"({df_funds.loc[fin_mask, 'Ticker'].nunique()} tickers kept in "
              f"universe).")

    print("[3/5] Aligning fundamentals to price history (merge_asof)...")
    df_prices = df_prices.sort_values('Date')
    df_funds = df_funds.sort_values('Filing_Date')

    master_df = pd.merge_asof(
        left=df_prices,
        right=df_funds,
        left_on='Date',
        right_on='Filing_Date',
        by='Ticker',
        direction='backward',
    )

    # set this from Ticker directly so pre-coverage rows get it too
    master_df['Is_Financial'] = master_df['Ticker'].isin(financial_tickers)

    # early rows with no previous filing get filled from the next filing for
    # that ticker. leaks up to ~6 weeks but the date window later drops most
    # of these rows
    n_pre_coverage = master_df['Filing_Date'].isna().sum()
    if n_pre_coverage > 0:
        master_df = master_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        fill_cols = (
            ['Filing_Date', 'fiscalDateEnding']
            + NUMERIC_COLS
            + ['eps_quarterly', 'fcf_quarterly',
            'eps_ttm', 'fcf_ttm', 'netIncome_ttm',
            'Gross_Margin', 'ROE', 'ROA', 'ROE_distorted', 'Current_Ratio']
        )

        # only apply the bfill to the pre-coverage rows. otherwise it fills in
        # values that were nulled on purpose (e.g. HD's distorted ROE) from much
        # later filings, which was the HD bug
        pre_coverage_mask = master_df['Filing_Date'].isna()
        for col in fill_cols:
            bfilled = master_df.groupby('Ticker')[col].bfill()
            master_df.loc[pre_coverage_mask, col] = bfilled.loc[pre_coverage_mask]

        n_filled = n_pre_coverage - master_df['Filing_Date'].isna().sum()
        print(f"   Forward-filled {n_filled:,} pre-coverage rows from next "
            f"available filing (bounded ~6wk lookahead).")

    n_before = len(master_df)
    master_df = master_df.dropna(subset=['Filing_Date'])
    n_no_coverage = n_before - len(master_df)
    if n_no_coverage > 0:
        print(f"   Dropped {n_no_coverage:,} rows with no fundamental "
              f"coverage for their ticker.")

    print(f"   Rows after merge: {len(master_df):,}")

    print("[4/5] Applying sanity filters on non-price ratios...")
    initial_len = len(master_df)

    valid_margin = (
        master_df['Gross_Margin'].isna()
        | master_df['Gross_Margin'].between(-10.0, 1.0)
    )
    # ROE was already capped above, this is just a backstop
    valid_roe = (
        master_df['ROE'].isna()
        | master_df['ROE'].between(-ROE_PLAUSIBILITY_CAP, ROE_PLAUSIBILITY_CAP)
    )
    # loose range for ROA, just to catch data errors
    valid_roa = (
        master_df['ROA'].isna()
        | master_df['ROA'].between(-1.0, 1.0)
    )
    valid_cr = (
        master_df['Current_Ratio'].isna()
        | master_df['Current_Ratio'].between(0.0, 100.0)
    )

    master_df = master_df[valid_margin & valid_roe & valid_roa & valid_cr]
    dropped = initial_len - len(master_df)
    print(f"   Dropped {dropped:,} rows on sanity filters "
          f"({dropped / initial_len * 100:.2f}%).")

    print("[5/5] Writing output...")
    # TTM inputs (P/E etc. get recomputed downstream) plus the non-price ratios
    output_cols = [
        'Date', 'Ticker', 'Filing_Date', 'fiscalDateEnding', 'Close',
        # TTM inputs for downstream P/E and FCF_Yield computation
        'eps_ttm',
        'fcf_ttm',
        'commonStockSharesOutstanding',
        # Pre-computed non-price ratios
        'Gross_Margin',
        'ROE',
        'ROA',
        'Current_Ratio',
        # Flags for downstream rendering / filtering
        'Is_Financial',
        'ROE_distorted',
    ]
    expert_features = master_df[output_cols]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    expert_features.to_parquet(OUTPUT_FILE, engine='pyarrow', index=False)

    n_with_filing = expert_features['Filing_Date'].notna().sum()
    pct = n_with_filing / len(expert_features) * 100 if len(expert_features) else 0
    n_eps_valid = expert_features['eps_ttm'].notna().sum()
    pct_eps = n_eps_valid / len(expert_features) * 100 if len(expert_features) else 0
    n_roa_valid = expert_features['ROA'].notna().sum()
    pct_roa = n_roa_valid / len(expert_features) * 100 if len(expert_features) else 0
    n_fin = int(expert_features['Is_Financial'].sum())

    print("-" * 55)
    print(f"SUCCESS - Fundamental Expert features saved")
    print(f"   Path:             {OUTPUT_FILE}")
    print(f"   Rows:             {len(expert_features):,}")
    print(f"   Tickers covered:  {expert_features['Ticker'].nunique()}")
    print(f"   Date range:       "
          f"{expert_features['Date'].min().date()} .. "
          f"{expert_features['Date'].max().date()}")
    print(f"   Filing_Date set:  {n_with_filing:,} / {len(expert_features):,} "
          f"({pct:.1f}%)")
    print(f"   TTM EPS set:      {n_eps_valid:,} / {len(expert_features):,} "
          f"({pct_eps:.1f}%)")
    print(f"   ROA set:          {n_roa_valid:,} / {len(expert_features):,} "
          f"({pct_roa:.1f}%)")
    print(f"   Financial rows:   {n_fin:,} (GM/Current_Ratio/FCF suppressed)")
    print(f"   Earnings lag:     +{EARNINGS_LAG_DAYS} days "
          f"(fiscalDateEnding -> Filing_Date)")
    print(f"   Note: P/E and FCF_Yield are computed downstream in "
          f"build_master_training_dataset.py")
    print(f"   Note: carry ROA + Is_Financial + ROE_distorted downstream and "
          f"render them in build_fundamental_input().")
    print("-" * 55)


if __name__ == "__main__":
    build_fundamental_features()