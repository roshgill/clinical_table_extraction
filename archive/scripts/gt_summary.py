"""Quick ground truth CSV summary — prints column count, row count, and column names."""

import sys
import pandas as pd


def summarize(path: str) -> None:
    df = pd.read_csv(path)
    print(f"File:    {path}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rows:    {len(df)}")
    print()
    for i, col in enumerate(df.columns):
        numeric = pd.to_numeric(df[col], errors="coerce")
        n_numeric = numeric.notna().sum()
        if n_numeric > 0:
            print(f"  [{i}] {col}: sum={numeric.sum():.4g} ({n_numeric}/{len(df)} numeric)")
        else:
            print(f"  [{i}] {col}: no numeric values")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raw = input("Path to CSV: ")
        paths = [raw.strip().strip("'\"")]
    else:
        paths = sys.argv[1:]
    for path in paths:
        summarize(path)
        if len(paths) > 1:
            print()
