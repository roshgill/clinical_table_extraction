"""
UniTable Confidence Score — Modal Script
=========================================
Runs all three UniTable passes (structure, bbox, cell) and records mean log
probability for each. Outputs three confidence signals per table:

  - conf_structure: mean log prob of structure tokens
  - conf_cell:      mean log prob of cell content tokens, averaged across cells
  - n_cells:        number of cells detected (bbox pass)

Usage:
    modal run scripts/confidence_score_modal.py::test   # 10 tables
    modal run scripts/confidence_score_modal.py::main   # 10,000 tables

Pull results:
    modal volume get pubtabnet-data /data/results/confidence_scores_<run_id>.csv ./confidence_scores.csv
"""

import modal
from pathlib import Path

ROOT              = Path(__file__).parent.parent
LOCAL_ICDAR_DIR   = ROOT / "Training set for encoder binary classifier pipeline" / "icdar-task-b" / "final_eval"
LOCAL_PUBTAB_DIR  = ROOT / "pubtabnet" / "images"
LOCAL_WEIGHTS_DIR = ROOT / "Huma-Huma" / "unitable" / "experiments" / "unitable_weights"
LOCAL_VOCAB_DIR   = ROOT / "Huma-Huma" / "unitable" / "vocab"

REMOTE_ICDAR_DIR   = "/images/icdar"
REMOTE_PUBTAB_DIR  = "/images/pubtabnet"
REMOTE_WEIGHTS_DIR = "/weights"
REMOTE_VOCAB_DIR   = "/vocab"

volume = modal.Volume.from_name("pubtabnet-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git"])
    .pip_install([
        "numpy<2",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "tokenizers==0.15.2",
        "einops",
        "Pillow",
        "pandas",
        "tqdm",
        "scikit-learn",
        "scipy",
        "jsonlines",
    ])
    .run_commands([
        "git clone https://github.com/poloclub/unitable.git /unitable",
        "cd /unitable && pip install -e . --quiet",
    ])
    .add_local_dir(str(LOCAL_ICDAR_DIR),   remote_path=REMOTE_ICDAR_DIR)
    .add_local_dir(str(LOCAL_PUBTAB_DIR),  remote_path=REMOTE_PUBTAB_DIR)
    .add_local_dir(str(LOCAL_WEIGHTS_DIR), remote_path=REMOTE_WEIGHTS_DIR)
    .add_local_dir(str(LOCAL_VOCAB_DIR),   remote_path=REMOTE_VOCAB_DIR)
)

app = modal.App("confidence-score-labeling", image=image)

N_WORKERS = 10


@app.function(
    gpu="A100",
    volumes={"/data": volume},
    timeout=86400,
    memory=32768,
)
def score_chunk(chunk: list, worker_id: int) -> list:
    import os, sys, re
    import torch
    import torch.nn as nn
    import tokenizers as tk
    from PIL import Image
    from torchvision import transforms
    from functools import partial
    from tqdm import tqdm

    sys.path.insert(0, "/unitable")
    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder
    from src.utils import (
        subsequent_mask, pred_token_within_range, greedy_sampling,
        bbox_str_to_token_list,
    )
    from src.vocab import HTML_TOKENS, BBOX_TOKENS, TASK_TOKENS, RESERVED_TOKENS

    VALID_HTML_TOKEN   = ["<eos>"] + HTML_TOKENS
    VALID_BBOX_TOKEN   = ["<eos>"] + BBOX_TOKENS
    INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {worker_id}] Device: {device}  Tables: {len(chunk)}")

    MEAN = [0.86597056, 0.88463002, 0.87491087]
    STD  = [0.20686628, 0.18201602, 0.18485524]
    D_MODEL, PATCH_SIZE, NHEAD, DROPOUT = 768, 16, 12, 0.2

    def make_model(vocab_path, max_seq_len, weights_path):
        vocab    = tk.Tokenizer.from_file(vocab_path)
        backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)
        encoder  = Encoder(d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT,
                           activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
        decoder  = Decoder(d_model=D_MODEL, nhead=NHEAD, dropout=DROPOUT,
                           activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)
        model = EncoderDecoder(
            backbone=backbone, encoder=encoder, decoder=decoder,
            vocab_size=vocab.get_vocab_size(), d_model=D_MODEL,
            padding_idx=vocab.token_to_id("<pad>"),
            max_seq_len=max_seq_len, dropout=DROPOUT,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return vocab, model.to(device).eval()

    vocab_html, model_html = make_model(
        f"{REMOTE_VOCAB_DIR}/vocab_html.json", 784,
        f"{REMOTE_WEIGHTS_DIR}/unitable_large_structure.pt"
    )
    vocab_bbox, model_bbox = make_model(
        f"{REMOTE_VOCAB_DIR}/vocab_bbox.json", 1024,
        f"{REMOTE_WEIGHTS_DIR}/unitable_large_bbox.pt"
    )
    vocab_cell, model_cell = make_model(
        f"{REMOTE_VOCAB_DIR}/vocab_cell_6k.json", 200,
        f"{REMOTE_WEIGHTS_DIR}/unitable_large_content.pt"
    )
    print(f"[Worker {worker_id}] All models loaded")

    transform_full = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    transform_cell = transforms.Compose([
        transforms.Resize((112, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def pad_to_square(image, fill=255):
        w, h = image.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
        padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        return padded

    def decode_with_confidence(model, img_tensor, prefix_id, eos_id,
                               whitelist=None, blacklist=None, max_len=512):
        """
        Greedy decode and return (token context, mean log prob of chosen tokens).
        Log prob is recorded from the FULL vocab distribution before whitelist
        filtering — same signal as the original structure-only script.
        """
        log_prob_sum = 0.0
        token_count  = 0
        with torch.no_grad():
            memory  = model.encode(img_tensor)
            context = torch.tensor([prefix_id], dtype=torch.int32)\
                           .repeat(img_tensor.shape[0], 1).to(device)
        for _ in range(max_len):
            if all(eos_id in row for row in context):
                break
            with torch.no_grad():
                mask   = subsequent_mask(context.shape[1]).to(device)
                logits = model.decode(memory, context, tgt_mask=mask, tgt_padding_mask=None)
                logits = model.generator(logits)[:, -1, :]
            log_probs       = torch.log_softmax(logits, dim=-1)
            logits_filtered = pred_token_within_range(
                logits.detach(), white_list=whitelist, black_list=blacklist
            )
            _, next_tok = torch.topk(logits_filtered, 1, dim=-1)
            # Accumulate log prob for each item in batch
            for b in range(img_tensor.shape[0]):
                chosen_id     = next_tok[b, 0].item()
                log_prob_sum += log_probs[b, chosen_id].item()
                token_count  += 1
            context = torch.cat([context, next_tok], dim=1)
        mean_lp = log_prob_sum / token_count if token_count > 0 else float("-inf")
        return context, mean_lp

    html_eos      = vocab_html.token_to_id("<eos>")
    html_prefix   = vocab_html.token_to_id("[html]")
    html_whitelist = [vocab_html.token_to_id(t) for t in VALID_HTML_TOKEN]

    bbox_eos      = vocab_bbox.token_to_id("<eos>")
    bbox_prefix   = vocab_bbox.token_to_id("[bbox]")
    bbox_whitelist = [vocab_bbox.token_to_id(t) for t in VALID_BBOX_TOKEN[:449]]

    cell_eos       = vocab_cell.token_to_id("<eos>")
    cell_prefix    = vocab_cell.token_to_id("[cell]")
    cell_blacklist = [vocab_cell.token_to_id(t) for t in INVALID_CELL_TOKEN]

    def score_table(image: Image.Image):
        """
        Returns (conf_structure, conf_cell, n_cells).
        conf_cell is the mean across all cells of each cell's mean log prob.
        Returns None for conf_cell if bbox pass finds no cells.
        """
        orig_w, orig_h = image.size
        padded   = pad_to_square(image)
        pad_size = padded.size[0]
        pad_x    = (pad_size - orig_w) // 2
        pad_y    = (pad_size - orig_h) // 2
        scale    = pad_size / 448
        img_t    = transform_full(padded).unsqueeze(0).to(device)

        # ── Structure pass ────────────────────────────────────────────────────
        _, conf_structure = decode_with_confidence(
            model_html, img_t, html_prefix, html_eos,
            whitelist=html_whitelist, max_len=512
        )

        # ── Bbox pass ─────────────────────────────────────────────────────────
        bbox_ctx, conf_bbox = decode_with_confidence(
            model_bbox, img_t, bbox_prefix, bbox_eos,
            whitelist=bbox_whitelist, max_len=1024
        )
        bbox_str = vocab_bbox.decode(
            bbox_ctx.detach().cpu().numpy()[0], skip_special_tokens=False
        )
        bboxes = bbox_str_to_token_list(bbox_str)
        if not bboxes:
            return conf_structure, conf_bbox, None, 0

        # Rescale bboxes back to original image coordinates
        bboxes_orig = [
            [max(0,      int(b[0] * scale) - pad_x),
             max(0,      int(b[1] * scale) - pad_y),
             min(orig_w, int(b[2] * scale) - pad_x),
             min(orig_h, int(b[3] * scale) - pad_y)]
            for b in bboxes
        ]

        # ── Cell content pass (batched) ───────────────────────────────────────
        cell_crops = []
        for b in bboxes_orig:
            crop = image.crop((b[0], b[1], b[2], b[3]))
            cell_crops.append(transform_cell(crop).to(device))

        cell_batch = torch.stack(cell_crops, dim=0)   # (n_cells, 3, 112, 448)

        _, conf_cell = decode_with_confidence(
            model_cell, cell_batch, cell_prefix, cell_eos,
            blacklist=cell_blacklist, max_len=200
        )

        return conf_structure, conf_bbox, conf_cell, len(bboxes_orig)

    import pandas as pd
    CHECKPOINT_EVERY = 200
    results  = []
    csv_path = f"/data/results/confidence_worker_{worker_id:02d}.csv"
    os.makedirs("/data/results", exist_ok=True)

    for row in tqdm(chunk, desc=f"Worker {worker_id}"):
        source   = row.get("source", "icdar")
        filename = row["filename"]
        img_path = (f"{REMOTE_PUBTAB_DIR}/{filename}" if source == "pubtabnet"
                    else f"{REMOTE_ICDAR_DIR}/{filename}")

        if not os.path.exists(img_path):
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            conf_structure, conf_bbox, conf_cell, n_cells = score_table(image)
            results.append({
                "filename":       filename,
                "source":         source,
                "teds_score":     row["teds_score"],
                "label":          row["label"],
                "conf_structure": conf_structure,
                "conf_bbox":      conf_bbox,
                "conf_cell":      conf_cell,   # None if no cells detected
                "n_cells":        n_cells,
            })
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR {filename}: {e}")

        if len(results) % CHECKPOINT_EVERY == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(csv_path, index=False)
            volume.commit()
            print(f"[Worker {worker_id}] Checkpoint — {len(results)} saved")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    volume.commit()
    print(f"[Worker {worker_id}] Done — {len(results)} rows → {csv_path}")
    return results


# ── Shared coordinator ────────────────────────────────────────────────────────

def _run(n_sample, tag):
    import math
    import pandas as pd
    from datetime import datetime

    SEED = 42

    df_icdar  = pd.read_csv(ROOT / "teds_labels_20260410_030718.csv")
    df_icdar["source"] = "icdar"
    df_pubtab = pd.read_csv(ROOT / "teds_labels_pubtabnet_full_20260411_200347.csv")
    df = pd.concat([df_icdar, df_pubtab], ignore_index=True)
    df_clean = df[df["label"].isin([0, 1]) & (df["teds_score"] >= 0)].copy()

    print(f"Combined clean dataset: {len(df_clean)} tables")

    n_pos = int(n_sample * (df_clean["label"] == 1).sum() / len(df_clean))
    n_neg = n_sample - n_pos
    sample_pos = df_clean[df_clean["label"] == 1].sample(
        n=min(n_pos, (df_clean["label"] == 1).sum()), random_state=SEED
    )
    sample_neg = df_clean[df_clean["label"] == 0].sample(
        n=min(n_neg, (df_clean["label"] == 0).sum()), random_state=SEED
    )
    df_sample = pd.concat([sample_pos, sample_neg]).reset_index(drop=True)
    print(f"Sample: {len(df_sample)}  (label=1: {(df_sample['label']==1).sum()}  label=0: {(df_sample['label']==0).sum()})")

    annotations = df_sample.to_dict("records")
    chunk_size  = math.ceil(len(annotations) / N_WORKERS)
    chunks      = [annotations[i:i + chunk_size] for i in range(0, len(annotations), chunk_size)]
    worker_ids  = list(range(len(chunks)))
    print(f"Workers: {len(chunks)}  Chunk size: {chunk_size}")

    all_results = []
    for chunk_results in score_chunk.starmap(zip(chunks, worker_ids)):
        all_results.extend(chunk_results)

    run_id   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/confidence_scores_{tag}_{run_id}.csv"
    df_out   = pd.DataFrame(all_results)
    df_out.to_csv(out_path, index=False)

    print(f"\nDone — {len(all_results)} tables processed")
    print(f"Saved locally: {out_path}")
    print(f"\nconf_structure stats:\n{df_out['conf_structure'].describe()}")
    print(f"\nconf_bbox stats:\n{df_out['conf_bbox'].describe()}")
    print(f"\nconf_cell stats:\n{df_out['conf_cell'].describe()}")
    print(f"\nn_cells stats:\n{df_out['n_cells'].describe()}")


@app.local_entrypoint()
def test():
    """Run on 10 tables with 1 worker — quick sanity check."""
    import pandas as pd
    from datetime import datetime

    SEED = 42
    df_icdar  = pd.read_csv(ROOT / "teds_labels_20260410_030718.csv")
    df_icdar["source"] = "icdar"
    df_pubtab = pd.read_csv(ROOT / "teds_labels_pubtabnet_full_20260411_200347.csv")
    df = pd.concat([df_icdar, df_pubtab], ignore_index=True)
    df_clean = df[df["label"].isin([0, 1]) & (df["teds_score"] >= 0)].copy()

    annotations = df_clean.sample(n=15, random_state=SEED).to_dict("records")
    print(f"Test run: 15 tables, 1 worker")

    all_results = list(score_chunk.remote(annotations, 0))

    run_id   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/confidence_scores_test_{run_id}.csv"
    df_out   = pd.DataFrame(all_results)
    df_out.to_csv(out_path, index=False)

    print(f"\nDone — {len(all_results)} tables processed")
    print(f"Saved locally: {out_path}")
    print(df_out[["filename", "source", "teds_score", "label",
                  "conf_structure", "conf_bbox", "conf_cell", "n_cells"]].to_string(index=False))


@app.local_entrypoint()
def main():
    """Run on 10,000 tables across 10 A100 workers."""
    _run(n_sample=10_000, tag="full")
