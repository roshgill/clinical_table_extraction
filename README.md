Extraction of tables from clinical PDF documents, with confidence-based routing to avoid sending difficult tables to a model that will produce bad output.

Pipeline
```
PDF page
  -> Table Transformer (detect table bounding boxes)
  -> UniTable structure decoder (extract HTML structure)
  -> Confidence score (mean log prob of structure tokens)
       >= threshold  ->  accept, return extracted table
       <  threshold  ->  route to human review
```

The confidence score is the mean log probability of each token chosen by UniTable's structure decoder during greedy decoding. Tables the model is uncertain about produce more negative scores and get filtered out before the full extraction runs.

The confidence score matches or beats a trained classifier with zero additional training. At the 100% precision threshold (`-0.00008`), no bad tables pass through. Recall at that point is ~2%, so the practical operating point trades some precision for more coverage. See `notebooks/research/04_confidence_threshold_analysis.ipynb` for the full sweep.

Repo Structure
```
agents/                 extraction agents (Gemini-based, early exploration)
notebooks/
  research/             numbered notebooks that tell the research story (start here)
  archive/              exploratory work, not part of the final pipeline
scripts/                Modal GPU scripts for large-scale data generation
data/                   generated CSVs (TEDS labels, confidence scores)
papers/                 input PDFs and ground truth CSVs
shared/                 shared utilities (PDF rendering, eval, Gemini client)
```

Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

UniTable weights are not included. Download from the [UniTable repo](https://github.com/poloclub/unitable) and place under `Huma-Huma/unitable/experiments/unitable_weights/`.
