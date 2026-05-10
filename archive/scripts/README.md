# Scripts

All scripts run on Modal (GPU cloud). Install Modal and authenticate before running any of them.

```bash
pip install modal
modal setup
```

## Data generation

**`unitable_teds_modal.py`** -- runs UniTable on the ICDAR 2021 eval set and scores each table with TEDS against ground truth. Produces `teds_labels_<run_id>.csv`.
```bash
modal run scripts/unitable_teds_modal.py
```

**`pubtabnet_teds_modal.py`** -- same as above but for PubTabNet. Requires `pubtabnet/images/` and `pubtabnet/annotations.jsonl` to exist locally (run `download_pubtabnet.py` first).
```bash
modal run scripts/pubtabnet_teds_modal.py
```

**`download_pubtabnet.py`** -- downloads PubTabNet images from HuggingFace and builds `annotations.jsonl`. Requires `HF_TOKEN` in `.env`.
```bash
python scripts/download_pubtabnet.py
```

## Confidence scoring

**`confidence_score_modal.py`** -- runs UniTable's structure decoder on a stratified sample and records the mean log probability per table. This is the core signal used for routing.
```bash
modal run scripts/confidence_score_modal.py::test   # 15 tables, sanity check
modal run scripts/confidence_score_modal.py::main   # 10,000 tables
```

## Classifier (explored, not used in final pipeline)

**`train_classifier_modal.py`** -- trains a binary classifier on frozen UniTable encoder features. ROC-AUC peaked at 0.66 across multiple architectures. Replaced by confidence scoring.
```bash
modal run scripts/train_classifier_modal.py
```
