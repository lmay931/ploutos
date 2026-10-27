"""
Formatting helpers shared between experts.

fmt is used by both the price/macro and fundamental experts so it lives here
rather than having two copies drift apart. Don't change the output format
without regenerating data, it ends up in the prompt text.
"""
import pandas as pd


def fmt(val, is_pct=False):
    if pd.isna(val):
        return "N/A"
    if is_pct:
        return f"{'+' if val > 0 else ''}{val * 100:.2f}%"
    return f"{val:.2f}"
