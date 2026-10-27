import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PipelineConfig

def generate_empirical_labels(input_csv, output_csv, forward_days=5,
                               flat_lower_pct=35.0, flat_upper_pct=65.0):
    """
    Calculates forward returns over forward_days and sets the thresholds from
    percentiles so the labels come out roughly 35% UP / 30% FLAT / 35% DOWN.
    """
    print(f"⏳ Loading data from {input_csv}...")
    
    # 1. Load Data (header=0 tells Pandas to skip the first row containing the text headers)
    column_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividend', 'Split', 'Ticker']
    df = pd.read_csv(input_csv, names=column_names, header=0)
    
    # 2. Convert to Datetime (errors='coerce' turns any rogue text rows into NaN)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 3. Drop any rows where the Date conversion failed (cleans up bad data/blank lines)
    df = df.dropna(subset=['Date'])
    
    # 4. Force numeric columns to be floats to fix the DtypeWarning
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    
    # 1. Calculate Forward Returns
    df[f'Close_{forward_days}d_Future'] = df.groupby('Ticker')['Close'].shift(-forward_days)
    df['Future_Return_Pct'] = (df[f'Close_{forward_days}d_Future'] - df['Close']) / df['Close']
    
    # Drop the NaN rows at the end of the dataset before calculating percentiles
    calc_df = df.dropna(subset=['Future_Return_Pct']).copy()
    
    # 2. Empirical check: Find the exact lower/upper percentiles
    lower_bound = np.percentile(calc_df['Future_Return_Pct'], flat_lower_pct)
    upper_bound = np.percentile(calc_df['Future_Return_Pct'], flat_upper_pct)
    
    print("\n[EMPIRICAL THRESHOLDS CALCULATED]")
    print(f"DOWN Threshold: {lower_bound*100:.2f}% or worse")
    print(f"UP Threshold:   {upper_bound*100:.2f}% or better")
    print(f"➖ FLAT Zone:      Between {lower_bound*100:.2f}% and {upper_bound*100:.2f}%\n")
    
    # 3. Apply the dynamic labels
    conditions = [
        calc_df['Future_Return_Pct'] > upper_bound,
        calc_df['Future_Return_Pct'] < lower_bound
    ]
    choices = ['UP', 'DOWN']
    calc_df['Target_Label'] = np.select(conditions, choices, default='FLAT')
    
    # 4. Clean up and Save
    calc_df['Future_Return_Pct'] = calc_df['Future_Return_Pct'].round(4)
    calc_df.to_csv(output_csv, index=False)
    
    print("-" * 50)
    print(f"Labeling Complete! Saved to: {output_csv}")
    print(f"Total Training Examples: {len(calc_df)}")
    print(f"Final Label Distribution:\n{calc_df['Target_Label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'}")
    print("-" * 50)

if __name__ == "__main__":
    cfg = PipelineConfig()
    input_file  = "data/raw_prices.csv"
    output_file = "data/labeled_training_data.csv"

    generate_empirical_labels(
        input_csv=input_file,
        output_csv=output_file,
        forward_days=cfg.horizon_days,
        flat_lower_pct=cfg.flat_lower_pct,
        flat_upper_pct=cfg.flat_upper_pct,
    )

    # quick check that the horizon setting actually got used
    import pandas as pd
    df = pd.read_csv(output_file, parse_dates=["Date"])
    print("\n=== HORIZON SANITY REPORT ===")
    print(f"  horizon_days      : {cfg.horizon_days}")
    print(f"  total rows        : {len(df)}")
    vc = df["Target_Label"].value_counts()
    for lbl in ("UP", "DOWN", "FLAT"):
        print(f"  {lbl:<6}            : {vc.get(lbl, 0)}")
    print(f"  usable date range : {df['Date'].min().date()} .. {df['Date'].max().date()}")
    print("=============================\n")