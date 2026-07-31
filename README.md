## Clinical Table Extraction Pipeline
Built for [HumaAI](https://www.qoniq.com/our-solution/) | Supervised by Eliot Levitt, CTO

HumaAI provides AI powered medical affairs systems for pharmaceutical clients. Medical and bio personnel query a knowledge base of clinical documents to retrieve precise statistical data. Errors in that data can directly affect patient outcomes.

This pipeline replaces a markdown-based ingestion system that sent unverified, malformed table data to downstream LLMs, causing confident but incorrect responses.

**Outcome:** 7,705 bad extractions rejected automatically across 9,026 tables. Zero incorrect extractions stored. ROC-AUC 0.8368 with no additional training, outperforming a purpose-trained classifier (0.58-0.66). 


### What It Does

Extracts tables from clinical PDF documents and routes them based on UniTable's own confidence in its output. Only tables where the model's decoder confidence exceeds threshold are stored for retrieval.

```
PDF page
  -> Table Transformer (Microsoft, trained on 1M+ tables) detect table regions
  -> UniTable (Georgia Tech, SOTA on 4/5 TR benchmarks) extract HTML structure
  -> Confidence filter (mean log prob of structure tokens)
       >= threshold  ->  parse into row/column cells, store for retrieval
       <  threshold  ->  route to human review
```

### Why Confidence Scoring

UniTable's structure decoder generates HTML one token at a time. At each step it assigns a probability to its chosen token. The mean log probability across all tokens is a direct measure of how certain the model was about the table it just read.

This signal requires zero additional training and outperforms a purpose-trained visual classifier:

| Method | ROC-AUC |
|--------|---------|
| Trained binary classifier (frozen UniTable encoder + head) | 0.58-0.66 |
| UniTable decoder confidence (mean log prob) | 0.8368 |

At the production threshold of -0.000010: 100% precision, zero bad tables passed through. 7,705 low-quality extractions correctly rejected across 9,026 test tables.

The threshold is conservative on purpose. This system serves medical teams. **Correctness matters more than coverage.**


### Results

- ROC-AUC: 0.8368 on 9,026 labeled tables (ICDAR 2021 + PubTabNet val)
- 100% precision at production threshold
- 7,705 bad tables rejected automatically
- Trained visual classifier explored and abandoned (ROC-AUC 0.58-0.66)

Full threshold sweep: notebooks/research/04_confidence_threshold_analysis.ipynb

### Future Work

This is research and validation work. The technique and threshold were established on 9,026 labeled tables from public benchmarks (ICDAR 2021 + PubTabNet val). The pipeline was handed off to the Qonic team for production integration and continued testing against their own documents.

Likely next steps from the team:
- Validation on Qonic's document set at production scale
- Threshold re-tuning if the distribution shifts on domain-specific documents
- Integration with the downstream LLM retrieval pipeline
- Monitoring for confidence-score drift over time as new document types appear

### Repo Structure

```
notebooks/research/    core findings (start here)
scripts/               Modal GPU scripts for data generation
data/                  generated CSVs (TEDS labels, confidence scores)
shared/                shared utilities
archive/               all prior exploration and dead ends
```

### Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

UniTable weights not included. Download from the [UniTable repo](https://github.com/poloclub/unitable) and place under `Huma-Huma/unitable/experiments/unitable_weights/`.
