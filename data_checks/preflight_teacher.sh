#!/usr/bin/env bash
# ============================================================
# checks to run before the teacher batch job
# run from the repo root, optionally pass the master csv path as the first arg
# ============================================================
set -uo pipefail
CSV="${1:-data/master_training_data.csv}"

if [[ ! -f "$CSV" ]]; then
  echo "ERROR: master CSV not found at '$CSV'."
  echo "usage: bash data_checks/preflight_teacher.sh path/to/master.csv"
  exit 1
fi

echo "Using master CSV: $CSV"
echo

echo "===== 1. encoding check (mojibake dashes) ====="
echo -n "mojibake dashes (should be 0): "
grep -c $'\xe2\x80\x9a\xc3\x84\xc3\xae' "$CSV" || true
echo -n "correct dashes (should be >0): "
grep -c $'\xe2\x80\x94' "$CSV" || true
echo "  if it's the other way round the file got mangled, back it up and fix with:"
echo "    cp \"$CSV\" \"$CSV.bak\""
echo "    python3 -c \"import pathlib;p=pathlib.Path('$CSV');p.write_text(p.read_text(encoding='utf-8').encode('latin-1').decode('utf-8'),encoding='utf-8')\""
echo

echo "===== 2. eligible rows, sentiment scoring, label balance ====="
CSV="$CSV" python3 - <<'PYEOF'
import os, pandas as pd
df = pd.read_csv(os.environ['CSV'])
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

elig = df[df['Target_Label'] != 'FLAT']
have_pm   = elig['Price_Macro_Signal'].notna()
have_fund = elig['Fund_Signal'].notna()
both_mask = have_pm & have_fund
teacher_eligible = int(both_mask.sum())

print(f'total rows:                 {len(df)}')
print(f'non-FLAT:                   {len(elig)}')
print(f'have price/macro signal:    {int(have_pm.sum())}')
print(f'have fundamental signal:    {int(have_fund.sum())}')
print(f'teacher eligible (pm and fund): {teacher_eligible}')
sent = elig.loc[both_mask, 'Sent_Signal']
print(f'   of which Sent NEUTRAL (still eligible):   {int((sent=="NEUTRAL").sum())}')
print(f'   of which Sent NaN (->NEUTRAL downstream): {int(sent.isna().sum())}')
print()

# check the llm actually scored the headlines and didn't just copy the old precomputed score
both = df.dropna(subset=['Sent_Agg_Score'])
print(f'Sent_Agg_Score populated:   {len(both)} rows')
stale_col = next((c for c in ['Sentiment_Score','Sentiment_Trend'] if c in df.columns), None)
if stale_col:
    s = both.dropna(subset=[stale_col])
    if len(s):
        identical = (s['Sent_Agg_Score'].round(2) == s[stale_col].round(2)).mean()
        print(f'same as {stale_col} (2dp): {identical:.1%}  '
              f'(above ~90% means it is probably copying)')
print()

print('Target_Label distribution (should be roughly balanced UP/ DOWN):')
print(df['Target_Label'].value_counts().to_string())
PYEOF
echo

echo "===== 3. extract_direction tests ====="
python3 -m pytest -q tests/test_verbalizer.py || echo "  extract_direction tests failing, fix before the teacher run"
echo

echo "If all of the above looks fine: empty data/sft_training_data.jsonl, run the"
echo "teacher with --mode create and --sample above the eligible count, and check"
echo "the '[BATCH] N rows eligible' line before submitting."
