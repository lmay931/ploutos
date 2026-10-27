"""
Checks Filing_Date in the master training CSV:
  1. no missing values
  2. parses as a datetime
  3. always <= Date (a later filing date means lookahead, probably from the bfill fallback)
  4. not too far behind Date (more than ~6 months means a quarter was missed)

Usage:
  python check_filing_dates.py [--csv path/to/master_training_data.csv]
"""

import argparse
import pandas as pd
import sys


def check_filing_dates(csv_path: str):
    print(f"[INFO] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df):,}, Tickers: {df['Ticker'].nunique()}")

    issues = []

    # =====================================================================
    # Check 1: Filing_Date is never missing
    # =====================================================================
    print("\n[CHECK 1] Filing_Date completeness")
    n_missing = df['Filing_Date'].isna().sum()
    if n_missing > 0:
        issues.append(f"{n_missing:,} rows have missing Filing_Date")
        print(f"   FAIL: {n_missing:,} rows missing Filing_Date")
        missing_per_ticker = (
            df[df['Filing_Date'].isna()]['Ticker']
            .value_counts()
            .head(10)
        )
        print(f"   Top tickers affected:")
        for tk, n in missing_per_ticker.items():
            print(f"      {tk}: {n}")
    else:
        print(f"   OK: all {len(df):,} rows have Filing_Date set")

    df_valid = df.dropna(subset=['Filing_Date']).copy()

    # =====================================================================
    # Check 2: Filing_Date parses as datetime
    # =====================================================================
    print("\n[CHECK 2] Filing_Date is parseable as datetime")
    try:
        df_valid['Filing_Date_parsed'] = pd.to_datetime(
            df_valid['Filing_Date'], errors='coerce'
        )
        df_valid['Date_parsed'] = pd.to_datetime(
            df_valid['Date'], errors='coerce'
        )
        n_unparseable = df_valid['Filing_Date_parsed'].isna().sum()
        if n_unparseable > 0:
            issues.append(f"{n_unparseable:,} Filing_Date values could not be parsed as datetime")
            print(f"   FAIL: {n_unparseable:,} rows have unparseable Filing_Date")
            bad_sample = df_valid[df_valid['Filing_Date_parsed'].isna()][
                ['Date', 'Ticker', 'Filing_Date']
            ].head(10)
            print(f"   Sample of unparseable rows:")
            print(bad_sample.to_string(index=False))
        else:
            print(f"   OK: all Filing_Date values parsed cleanly")
    except Exception as e:
        issues.append(f"Parse error: {e}")
        print(f"   FAIL: parse error: {e}")
        return issues

    df_valid = df_valid.dropna(subset=['Filing_Date_parsed', 'Date_parsed'])

    # =====================================================================
    # Check 3: Filing_Date <= Date (no lookahead bias)
    # =====================================================================
    print("\n[CHECK 3] Filing_Date <= Date (no future filings)")
    future_filings = df_valid['Filing_Date_parsed'] > df_valid['Date_parsed']
    n_future = future_filings.sum()
    if n_future > 0:
        issues.append(f"{n_future:,} rows have Filing_Date AFTER Date (lookahead bias)")
        print(f"   FAIL: {n_future:,} rows have Filing_Date > Date")
        print(f"   This indicates the bfill fallback (for pre-coverage rows) "
              f"or some other ffill bug is creating lookahead bias.")
        future_per_ticker = (
            df_valid[future_filings]['Ticker'].value_counts().head(10)
        )
        print(f"   Top tickers affected:")
        for tk, n in future_per_ticker.items():
            print(f"      {tk}: {n}")
        sample = df_valid[future_filings][
            ['Date', 'Ticker', 'Filing_Date']
        ].sort_values('Date').head(10)
        print(f"   Sample rows (sorted by Date):")
        print(sample.to_string(index=False))
    else:
        print(f"   OK: all Filing_Date values are <= Date")

    # =====================================================================
    # Check 4: Filing_Date is not unreasonably stale
    # =====================================================================
    print("\n[CHECK 4] Filing_Date staleness vs Date")
    df_valid['days_stale'] = (
        df_valid['Date_parsed'] - df_valid['Filing_Date_parsed']
    ).dt.days

    # quarters are ~91 days apart so with the 45 day filing lag it shouldn't get much
    # past one quarter plus a bit for late filers. 200+ days behind means a quarter is missing
    STALENESS_THRESHOLD_DAYS = 200

    print(f"   Days-stale distribution (Date - Filing_Date):")
    print(f"      min:    {df_valid['days_stale'].min():.0f}")
    print(f"      median: {df_valid['days_stale'].median():.0f}")
    print(f"      mean:   {df_valid['days_stale'].mean():.1f}")
    print(f"      p90:    {df_valid['days_stale'].quantile(0.9):.0f}")
    print(f"      p99:    {df_valid['days_stale'].quantile(0.99):.0f}")
    print(f"      max:    {df_valid['days_stale'].max():.0f}")

    n_stale = (df_valid['days_stale'] > STALENESS_THRESHOLD_DAYS).sum()
    if n_stale > 0:
        issues.append(f"{n_stale:,} rows have Filing_Date >{STALENESS_THRESHOLD_DAYS} days behind Date (missed quarter?)")
        print(f"   WARN: {n_stale:,} rows ({n_stale/len(df_valid)*100:.1f}%) "
              f"have Filing_Date >{STALENESS_THRESHOLD_DAYS} days behind Date")
        stale_per_ticker = (
            df_valid[df_valid['days_stale'] > STALENESS_THRESHOLD_DAYS]
            ['Ticker'].value_counts().head(10)
        )
        print(f"   Top tickers with stale filings:")
        for tk, n in stale_per_ticker.items():
            print(f"      {tk}: {n}")
    else:
        print(f"   OK: all Filing_Date values are within "
              f"{STALENESS_THRESHOLD_DAYS} days of Date")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 60}")
    if not issues:
        print(f"SUCCESS - all Filing_Date integrity checks passed.")
        print(f"{'=' * 60}")
        return 0
    else:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print(f"{'=' * 60}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv',
        default='data/master_training_data.csv',
        help='Path to the master training CSV.',
    )
    args = parser.parse_args()
    sys.exit(check_filing_dates(args.csv))
