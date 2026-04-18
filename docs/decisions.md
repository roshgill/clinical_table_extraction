# Key Decisions

## Why UniTable over Gemini for table extraction

Gemini (vision LLM) was the first approach. Evals on ICDAR and clinical tables showed inconsistent structure -- column merging errors, missed rows, hallucinated content. UniTable, trained specifically on table structure recognition, produced more reliable HTML output and runs locally without per-call API cost.

## Why confidence score over a trained classifier

A binary classifier (good table / bad table) was trained on frozen UniTable encoder features across ICDAR + PubTabNet (17,000+ tables). Multiple architectures and training configurations were tested. ROC-AUC plateaued at 0.58-0.66 -- barely above random.

The root cause: the frozen encoder produces features useful for table reconstruction, not for difficulty prediction. The signal simply is not there in the encoder representation.

The confidence score (mean log probability of structure decoder tokens) requires no training. It uses UniTable's own uncertainty signal directly. On 10,000 tables it achieved ROC-AUC 0.57 with a clean semantic interpretation: low confidence means the model cannot confidently parse the table structure, so the output will likely be wrong.

## Why structure decoder only (not bbox or cell passes)

All three UniTable passes were tested. Pearson correlation with TEDS on 15 tables:
- Structure: 0.57
- Bbox: -0.01
- Cell: -0.17

Bbox and cell signals are noisy and do not improve on structure alone. The structure pass is also the fastest -- no cell crops needed.

## Why bad crops from Table Transformer do not need a separate VLM filter

A crop that is not a real table (caption, partial table, figure) will produce an uncertain structure decode -- the model has no schema to confidently assign. It will score low on the confidence filter and be routed to human review automatically. A dedicated VLM pre-filter adds latency and cost for no incremental benefit.

## Confidence threshold operating point

At `mean_log_prob >= -0.00001`: 100% precision, ~2% recall. Almost nothing passes through.

In practice, choose a threshold based on the acceptable false-positive rate for the deployment context. The full sweep is in `notebooks/research/04_confidence_threshold_analysis.ipynb`.
