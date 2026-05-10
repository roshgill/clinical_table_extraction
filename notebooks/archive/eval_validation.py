"""
Demo: Table Extraction Validation Pipeline
-------------------------------------------
Shows how the failure detection layer integrates with:
  1. UniTable/Gemini extraction pipeline
  2. PubTabNet ground truth evaluation
  3. The iterative rule-refinement loop

This is the layer between extraction and downstream storage.
"""

import json
from shared.table_validator import (
    validate_table, validate_against_ground_truth,
    validate_batch, batch_summary,
    Confidence, FailureType
)


# -- Simulated extraction outputs (in practice, these come from the eval notebook) --
# Using realistic medical table HTML to demonstrate the validator

SAMPLE_EXTRACTIONS = {
    # Clean extraction -- statin comparison table
    "lipitor_table3.png": """<html><body><table>
<thead>
<tr><th>Treatment</th><th>N</th><th>LDL-C Change (%)</th><th>p-value</th></tr>
</thead>
<tbody>
<tr><td>Atorvastatin 10mg</td><td>257</td><td>-39.0</td><td>&lt;0.001</td></tr>
<tr><td>Atorvastatin 20mg</td><td>260</td><td>-43.2</td><td>&lt;0.001</td></tr>
<tr><td>Atorvastatin 40mg</td><td>248</td><td>-50.1</td><td>&lt;0.001</td></tr>
<tr><td>Placebo</td><td>252</td><td>-1.2</td><td>--</td></tr>
</tbody>
</table></body></html>""",

    # Partially broken -- missing header, column mismatch
    "ctt_meta_table1.png": """<html><body><table>
<tr><td>Trial</td><td>Events/Statin</td><td>Events/Control</td><td>RR (95% CI)</td></tr>
<tr><td>4S</td><td>182/2221</td><td>256/2223</td><td>0.71 (0.59-0.85)</td></tr>
<tr><td>WOSCOPS</td><td>174/3302</td><td>248/3293</td></tr>
<tr><td>CARE</td><td>212/2081</td><td>274/2078</td><td>0.76 (0.64-0.91)</td></tr>
<tr><td>LIPID</td><td>557/4512</td><td>715/4502</td><td>0.78 (0.69-0.87)</td></tr>
</table></body></html>""",

    # OCR disaster -- broken numbers, empty cells, no structure
    "hodkinson_table2.png": """<html><body><table>
<tr><td></td><td>Outcome</td><td></td></tr>
<tr><td>Study</td><td>HR</td><td>95% Cl</td></tr>
<tr><td>Smith 20l9</td><td>0 . 82</td><td>0.7l - 0.94</td></tr>
<tr><td>Jones 202O</td><td>l.05</td><td>0.88 - l.25</td></tr>
<tr><td></td><td></td><td></td></tr>
<tr><td>Overall</td><td>0.9l</td><td>0.84-0.99</td></tr>
</table></body></html>""",

    # Spanning table -- FDA review style, complex headers
    "lipitor_review_table8.png": """<html><body><table>
<thead>
<tr><th rowspan="2">Parameter</th><th colspan="3">Atorvastatin</th><th colspan="3">Placebo</th></tr>
<tr><th>N</th><th>Mean</th><th>SD</th><th>N</th><th>Mean</th><th>SD</th></tr>
</thead>
<tbody>
<tr><td>Total Cholesterol (mg/dL)</td><td>127</td><td>198.4</td><td>32.1</td><td>131</td><td>263.2</td><td>35.8</td></tr>
<tr><td>LDL-C (mg/dL)</td><td>127</td><td>112.6</td><td>28.4</td><td>131</td><td>178.9</td><td>30.2</td></tr>
<tr><td>HDL-C (mg/dL)</td><td>127</td><td>52.1</td><td>14.2</td><td>131</td><td>48.7</td><td>12.8</td></tr>
<tr><td>Triglycerides (mg/dL)</td><td>127</td><td>141.3</td><td>68.5</td><td>131</td><td>182.4</td><td>74.1</td></tr>
</tbody>
</table></body></html>""",

    # Single column disaster
    "geodan_table5.png": """<html><body><table>
<tr><td>Adverse Event: Headache - 12% vs 8%</td></tr>
<tr><td>Adverse Event: Nausea - 8% vs 5%</td></tr>
<tr><td>Adverse Event: Dizziness - 6% vs 4%</td></tr>
<tr><td>Adverse Event: Insomnia - 4% vs 3%</td></tr>
</table></body></html>""",
}

# ground truth for one of them (simulating XML-derived GT)
GROUND_TRUTH = {
    "lipitor_table3.png": """<html><body><table>
<thead>
<tr><th>Treatment</th><th>N</th><th>LDL-C Change (%)</th><th>p-value</th></tr>
</thead>
<tbody>
<tr><td>Atorvastatin 10 mg</td><td>257</td><td>-39.0</td><td>&lt;0.001</td></tr>
<tr><td>Atorvastatin 20 mg</td><td>260</td><td>-43.2</td><td>&lt;0.001</td></tr>
<tr><td>Atorvastatin 40 mg</td><td>248</td><td>-50.1</td><td>&lt;0.001</td></tr>
<tr><td>Placebo</td><td>252</td><td>-1.2</td><td>--</td></tr>
</tbody>
</table></body></html>""",
}


def main():
    print("=" * 70)
    print("TABLE EXTRACTION VALIDATION PIPELINE")
    print("=" * 70)

    # Step 1: validate all extractions
    print("\n--- Step 1: Structural validation (no ground truth needed) ---\n")
    results = validate_batch(SAMPLE_EXTRACTIONS)
    print(batch_summary(results))

    # Step 2: ground truth comparison where available
    print("\n--- Step 2: Ground truth comparison (XML-derived) ---\n")
    for fname, gt_html in GROUND_TRUTH.items():
        pred_html = SAMPLE_EXTRACTIONS.get(fname, "")
        if pred_html:
            diff = validate_against_ground_truth(pred_html, gt_html)
            print(f"{fname}:")
            print(f"  rows match: {diff['row_count_match']}")
            print(f"  col structure match: {diff['col_structure_match']}")
            print(f"  cell accuracy: {diff['cell_accuracy']:.1%}")
            print(f"  failure modes: {diff['failure_modes']}")
            if diff['mismatch_examples']:
                print(f"  sample mismatches:")
                for ex in diff['mismatch_examples'][:3]:
                    print(f"    [{ex['row']},{ex['col']}] "
                          f"pred='{ex['predicted']}' gt='{ex['ground_truth']}'")
            print()

    # Step 3: failure mode aggregation (feeds into rule refinement)
    print("\n--- Step 3: Failure mode summary (for rule refinement loop) ---\n")
    failure_counts = {}
    for fname, res in results.items():
        for issue in res.issues:
            ft = issue.failure_type.value
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

    if failure_counts:
        for ft, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {ft}: {count} occurrence(s)")
    else:
        print("  no failures detected")

    # Step 4: routing decisions
    print("\n--- Step 4: Pipeline routing decisions ---\n")
    for fname, res in results.items():
        action = {
            Confidence.HIGH: "-> store directly, ready for RAG",
            Confidence.MEDIUM: "-> route to Gemini refinement pass",
            Confidence.LOW: "-> flag for manual review or re-extraction",
        }[res.confidence]
        print(f"  {fname}: [{res.confidence.value.upper()}] {action}")


if __name__ == "__main__":
    main()