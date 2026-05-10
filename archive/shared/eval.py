"""Accuracy comparison utilities — shared across all agents."""

import pandas as pd
from IPython.display import display


def normalize_text(text) -> str:
    """Normalize text for comparison: strip whitespace, collapse spaces, remove commas/periods, lowercase."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('\u00b7', '.')   # middle dot → period
    text = text.replace('\u2013', '-')   # en-dash → hyphen
    text = text.replace('\u2212', '-')   # minus sign → hyphen
    text = text.replace(",", "").replace(".", "").lower()
    return " ".join(text.split())


def compare_tables(df_truth: pd.DataFrame, df_extracted: pd.DataFrame) -> dict:
    """Compare extracted table against ground truth by row and column position.

    Returns a dict with accuracy stats, mismatches list, and per-column breakdown.
    """
    truth_rows = len(df_truth)
    extracted_rows = len(df_extracted)
    compare_rows = min(truth_rows, extracted_rows)
    compare_cols = min(len(df_truth.columns), len(df_extracted.columns))

    total_cells = 0
    correct_cells = 0
    mismatches = []
    per_column_stats = {}

    for col_idx in range(compare_cols):
        truth_col = df_truth.columns[col_idx]
        col_label = normalize_text(truth_col)
        per_column_stats[col_label] = {"total": 0, "correct": 0}

        for row_idx in range(compare_rows):
            truth_val = normalize_text(df_truth.iloc[row_idx, col_idx])
            extracted_val = normalize_text(df_extracted.iloc[row_idx, col_idx])

            total_cells += 1
            per_column_stats[col_label]["total"] += 1

            if truth_val == extracted_val:
                correct_cells += 1
                per_column_stats[col_label]["correct"] += 1
            else:
                mismatches.append({
                    "Row": row_idx,
                    "Column": col_label,
                    "Ground Truth": truth_val,
                    "Extracted": extracted_val,
                })

    # Row-level match: fraction of rows where ALL cells match
    rows_fully_correct = 0
    for row_idx in range(compare_rows):
        row_correct = all(
            normalize_text(df_truth.iloc[row_idx, col_idx]) == normalize_text(df_extracted.iloc[row_idx, col_idx])
            for col_idx in range(compare_cols)
        )
        if row_correct:
            rows_fully_correct += 1

    return {
        "truth_count": truth_rows,
        "extracted_count": extracted_rows,
        "rows_compared": compare_rows,
        "missing_rows": max(0, truth_rows - extracted_rows),
        "extra_rows": max(0, extracted_rows - truth_rows),
        "total_cells": total_cells,
        "correct_cells": correct_cells,
        "cell_accuracy": correct_cells / total_cells if total_cells > 0 else 0,
        "row_match": rows_fully_correct / compare_rows if compare_rows > 0 else 0,
        "rows_fully_correct": rows_fully_correct,
        "mismatches": mismatches,
        "per_column_stats": per_column_stats,
    }


def print_accuracy_summary(results: dict):
    """Print a formatted accuracy summary from compare_tables() output."""
    print("=" * 60)
    print("ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Ground truth rows:  {results['truth_count']}")
    print(f"Extracted rows:     {results['extracted_count']}")
    print(f"Rows compared:      {results['rows_compared']}")
    print(f"Missing rows:       {results['missing_rows']}")
    print(f"Extra rows:         {results['extra_rows']}")
    print()
    print(f"Total cells compared: {results['total_cells']}")
    print(f"Correct cells:        {results['correct_cells']}")
    print(f"Cell accuracy:        {results['cell_accuracy']:.1%}")
    print(f"Row match:            {results['row_match']:.1%}  ({results['rows_fully_correct']}/{results['rows_compared']} rows fully correct)")

    if results["mismatches"]:
        print(f"\n{'=' * 60}")
        print(f"DETAILED MISMATCHES ({len(results['mismatches'])} cells)")
        print(f"{'=' * 60}")
        df_mismatches = pd.DataFrame(results["mismatches"])
        display(df_mismatches)
    else:
        print("\nPerfect match! No mismatches found.")

    # Per-column accuracy breakdown
    print(f"\n{'=' * 60}")
    print("PER-COLUMN ACCURACY")
    print(f"{'=' * 60}")

    col_data = []
    for col, stats in results["per_column_stats"].items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        col_data.append({
            "Column": col,
            "Total Cells": stats["total"],
            "Correct": stats["correct"],
            "Errors": stats["total"] - stats["correct"],
            "Accuracy": f"{accuracy:.1%}",
        })

    df_col_accuracy = pd.DataFrame(col_data)
    display(df_col_accuracy)
