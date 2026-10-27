import pandas as pd

csv_path = "data/master_training_data.csv"

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# dates got saved day first so parse them that way
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

if 'Filing_Date' in df.columns:
    df['Filing_Date'] = pd.to_datetime(df['Filing_Date'], dayfirst=True, errors='coerce')
if 'fiscalDateEnding' in df.columns:
    df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'], dayfirst=True, errors='coerce')

# overwrite the csv with YYYY-MM-DD dates
df.to_csv(csv_path, index=False, date_format='%Y-%m-%d')
print("Dates restored and saved successfully!")