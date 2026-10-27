import pandas as pd
import os

DATA_DIR = 'data'
PRICE_FILE = 'raw_prices.csv'
# ---- quick check of the date formats in raw_prices.csv ----


df_prices = pd.read_csv(os.path.join(DATA_DIR, PRICE_FILE))

print(df_prices.head(10))  # Print the first 10 date entries to inspect their format
# 1. parse with the expected format, anything that doesn't match becomes NaT
coerced_dates = pd.to_datetime(df_prices['Date'], format="%Y-%m-%d", errors='coerce')

# 2. Find rows that failed to parse but are not originally empty/null
invalid_dates = df_prices.loc[coerced_dates.isna() & df_prices['Date'].notna(), 'Date']

# 3. Print the total count of non-matching rows
print(f"\n[INSPECTION] Total rows not matching '%d/%m/%Y': {len(invalid_dates)}")

# 4. Print the unique values to see what other formats exist (e.g., YYYY-MM-DD)
print("[INSPECTION] Unique non-matching date strings found:")
print(invalid_dates.unique())

# 5. Print the first 20 occurrences with their DataFrame index positions
print("\n[INSPECTION] First 20 non-matching dates:")
print(invalid_dates.head(20))

# -----------------------------------------------------------------