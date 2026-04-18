"""
UniTable TEDS Label Generation — Modal Script
=============================================
Runs UniTable inference on local ICDAR 2021 test set in parallel across
multiple GPU workers. Each worker loads models once and processes a chunk
of tables, then results are merged and saved to the volume.

Usage:
    modal run scripts/unitable_teds_modal.py

Pull results locally after completion:
    modal volume get pubtabnet-data /data/results/teds_labels_<run_id>.csv ./teds_labels.csv
    modal volume ls pubtabnet-data /data/results/
"""

import modal
from pathlib import Path

# ── Local paths ───────────────────────────────────────────────────────────────
LOCAL_ICDAR_DIR    = Path(__file__).parent.parent / "Training set for encoder binary classifier pipeline" / "icdar-task-b"
LOCAL_WEIGHTS_DIR  = Path(__file__).parent.parent / "Huma-Huma" / "unitable" / "experiments" / "unitable_weights"
LOCAL_VOCAB_DIR    = Path(__file__).parent.parent / "Huma-Huma" / "unitable" / "vocab"

# ── Remote paths ──────────────────────────────────────────────────────────────
REMOTE_ICDAR_DIR   = "/icdar"
REMOTE_WEIGHTS_DIR = "/weights"
REMOTE_VOCAB_DIR   = "/vocab"
RESULTS_DIR        = "/data/results"

# ── Volume for results ────────────────────────────────────────────────────────
volume = modal.Volume.from_name("pubtabnet-data", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git"])
    .pip_install([
        "numpy<2",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "tokenizers==0.15.2",
        "einops",
        "jsonlines",
        "beautifulsoup4",
        "hydra-core",
        "hydra_colorlog",
        "apted",
        "Distance",
        "lxml==4.9.3",
        "torchmetrics",
        "Pillow",
        "pandas",
        "tqdm",
    ])
    .run_commands([
        "git clone https://github.com/poloclub/unitable.git /unitable",
        "cd /unitable && pip install -e . --quiet",
    ])
    .add_local_dir(str(LOCAL_ICDAR_DIR),   remote_path=REMOTE_ICDAR_DIR)
    .add_local_dir(str(LOCAL_WEIGHTS_DIR), remote_path=REMOTE_WEIGHTS_DIR)
    .add_local_dir(str(LOCAL_VOCAB_DIR),   remote_path=REMOTE_VOCAB_DIR)
)

app = modal.App("unitable-teds-labeling", image=image)

# ── Config ────────────────────────────────────────────────────────────────────
LABEL_HIGH  = 0.95   # TEDS >= this → label 1 (positive)
LABEL_LOW   = 0.85   # TEDS <= this → label 0 (negative)
N_TABLES    = None   # None = all; set to int to cap for testing
N_WORKERS   = 10     # parallel GPU workers (match your GPU limit)


# ── Per-worker function ───────────────────────────────────────────────────────
@app.function(
    gpu="A100",
    volumes={"/data": volume},
    timeout=86400,
    memory=32768,
)
def process_chunk(chunk: list, worker_id: int) -> list:
    """Load models once, process all annotations in this chunk, return results."""
    import os, re, sys, json
    import torch
    import tokenizers as tk
    from PIL import Image
    from torchvision import transforms
    from torch import nn, Tensor
    from functools import partial
    from tqdm import tqdm

    sys.path.insert(0, "/unitable")
    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder  # type: ignore
    from src.utils import (  # type: ignore
        subsequent_mask, pred_token_within_range, greedy_sampling,
        bbox_str_to_token_list, cell_str_to_token_list,
        html_str_to_token_list, build_table_from_html_and_cell,
        html_table_template,
    )
    from src.vocab import HTML_TOKENS, TASK_TOKENS, RESERVED_TOKENS, BBOX_TOKENS  # type: ignore
    VALID_HTML_TOKEN   = ["<eos>"] + HTML_TOKENS
    VALID_BBOX_TOKEN   = ["<eos>"] + BBOX_TOKENS
    INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
    from src.utils.teds import TEDS  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {worker_id}] Device: {device}  Tables: {len(chunk)}")

    # ── Load models ────────────────────────────────────────────────────────────
    MEAN = [0.86597056, 0.88463002, 0.87491087]
    STD  = [0.20686628, 0.18201602, 0.18485524]
    d_model, patch_size, nhead, dropout = 768, 16, 12, 0.2

    def make_model(vocab_path, max_seq_len, weights_path):
        vocab    = tk.Tokenizer.from_file(str(vocab_path))
        backbone = ImgLinearBackbone(d_model=d_model, patch_size=patch_size)
        encoder  = Encoder(d_model=d_model, nhead=nhead, dropout=dropout,
                           activation="gelu", norm_first=True, nlayer=12, ff_ratio=4)
        decoder  = Decoder(d_model=d_model, nhead=nhead, dropout=dropout,
                           activation="gelu", norm_first=True, nlayer=4, ff_ratio=4)
        model = EncoderDecoder(
            backbone=backbone, encoder=encoder, decoder=decoder,
            vocab_size=vocab.get_vocab_size(), d_model=d_model,
            padding_idx=vocab.token_to_id("<pad>"),
            max_seq_len=max_seq_len, dropout=dropout,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return vocab, model.to(device).eval()

    vocab_html, model_html = make_model(f"{REMOTE_VOCAB_DIR}/vocab_html.json",    784,  f"{REMOTE_WEIGHTS_DIR}/unitable_large_structure.pt")
    vocab_bbox, model_bbox = make_model(f"{REMOTE_VOCAB_DIR}/vocab_bbox.json",    1024, f"{REMOTE_WEIGHTS_DIR}/unitable_large_bbox.pt")
    vocab_cell, model_cell = make_model(f"{REMOTE_VOCAB_DIR}/vocab_cell_6k.json", 200,  f"{REMOTE_WEIGHTS_DIR}/unitable_large_content.pt")
    print(f"[Worker {worker_id}] Models loaded")

    # ── Inference helpers ──────────────────────────────────────────────────────
    def pad_to_square(image: Image.Image, fill: int = 255) -> Image.Image:
        w, h    = image.size
        max_dim = max(w, h)
        padded  = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
        padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        return padded

    def to_tensor(image: Image.Image, size=(448, 448)) -> Tensor:
        T = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return T(image).to(device).unsqueeze(0)  # type: ignore[union-attr]

    def decode(model, image: Tensor, prefix, max_len, eos_id,
               whitelist=None, blacklist=None) -> Tensor:
        with torch.no_grad():
            memory  = model.encode(image)
            context = torch.tensor(prefix, dtype=torch.int32)\
                           .repeat(image.shape[0], 1).to(device)
        for _ in range(max_len):
            if all(eos_id in k for k in context):
                break
            with torch.no_grad():
                mask   = subsequent_mask(context.shape[1]).to(device)
                logits = model.decode(memory, context,
                                      tgt_mask=mask, tgt_padding_mask=None)
                logits = model.generator(logits)[:, -1, :]
            logits = pred_token_within_range(
                logits.detach(), white_list=whitelist, black_list=blacklist
            )
            _, next_tok = greedy_sampling(logits)
            context = torch.cat([context, next_tok], dim=1)
        return context

    def run_unitable(image: Image.Image):
        orig_w, orig_h = image.size
        padded   = pad_to_square(image)
        pad_size = padded.size[0]
        pad_x    = (pad_size - orig_w) // 2
        pad_y    = (pad_size - orig_h) // 2
        scale    = pad_size / 448

        img_t = to_tensor(padded, (448, 448))

        ph = decode(model_html, img_t,
                    prefix=[vocab_html.token_to_id("[html]")],
                    max_len=512, eos_id=vocab_html.token_to_id("<eos>"),
                    whitelist=[vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN])
        ph = vocab_html.decode(ph.detach().cpu().numpy()[0], skip_special_tokens=False)
        ph = html_str_to_token_list(ph)

        pb = decode(model_bbox, img_t,
                    prefix=[vocab_bbox.token_to_id("[bbox]")],
                    max_len=1024, eos_id=vocab_bbox.token_to_id("<eos>"),
                    whitelist=[vocab_bbox.token_to_id(i) for i in VALID_BBOX_TOKEN[:449]])
        pb = vocab_bbox.decode(pb.detach().cpu().numpy()[0], skip_special_tokens=False)
        pb = bbox_str_to_token_list(pb)

        if not pb:
            return None

        pb_orig = [
            [
                max(0,      int(b[0] * scale) - pad_x),
                max(0,      int(b[1] * scale) - pad_y),
                min(orig_w, int(b[2] * scale) - pad_x),
                min(orig_h, int(b[3] * scale) - pad_y),
            ]
            for b in pb
        ]

        cell_t = torch.cat(
            [to_tensor(image.crop((b[0], b[1], b[2], b[3])), (112, 448)) for b in pb_orig], dim=0
        )
        pc = decode(model_cell, cell_t,
                    prefix=[vocab_cell.token_to_id("[cell]")],
                    max_len=200, eos_id=vocab_cell.token_to_id("<eos>"),
                    blacklist=[vocab_cell.token_to_id(i) for i in INVALID_CELL_TOKEN])
        pc = vocab_cell.decode_batch(pc.detach().cpu().numpy(), skip_special_tokens=False)
        pc = [cell_str_to_token_list(i) for i in pc]
        pc = [re.sub(r"(\d).\s+(\d)", r"\1.\2", i) for i in pc]

        pred_code = build_table_from_html_and_cell(ph, pc)
        return html_table_template("".join(pred_code))

    # ── Process chunk ──────────────────────────────────────────────────────────
    import pandas as pd
    CHECKPOINT_EVERY = 500
    teds_metric  = TEDS(structure_only=False, ignore_nodes="b")
    results      = []   # rows for labels CSV
    html_records = []   # rows for full HTML JSONL
    csv_path     = f"{RESULTS_DIR}/worker_{worker_id:02d}.csv"
    jsonl_path   = f"{RESULTS_DIR}/worker_{worker_id:02d}.jsonl"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for i, anno in enumerate(tqdm(chunk, desc=f"Worker {worker_id}")):
        fname    = anno["filename"]
        img_path = f"{REMOTE_ICDAR_DIR}/final_eval/{fname}"

        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            pred  = run_unitable(image)
            score = teds_metric.evaluate(pred, anno["gt_html"]) if pred else 0.0
            label = 1 if score >= LABEL_HIGH else 0 if score <= LABEL_LOW else -1

            results.append({
                "filename":   fname,
                "type":       anno["type"],
                "teds_score": round(score, 4),
                "label":      label,
            })
            html_records.append({
                "filename":   fname,
                "teds_score": round(score, 4),
                "pred_html":  pred or "",
                "gt_html":    anno["gt_html"],
            })
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR {fname}: {e}")

        if len(results) % CHECKPOINT_EVERY == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(csv_path, index=False)
            with open(jsonl_path, "w") as f:
                for rec in html_records:
                    f.write(json.dumps(rec) + "\n")
            volume.commit()
            print(f"[Worker {worker_id}] Checkpoint — {len(results)} saved to volume")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    with open(jsonl_path, "w") as f:
        for rec in html_records:
            f.write(json.dumps(rec) + "\n")
    volume.commit()
    print(f"[Worker {worker_id}] Done — {len(results)} processed → {csv_path} + {jsonl_path}")
    return results


# ── Coordinator ───────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import json, math
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    # Load and optionally cap annotations locally
    with open(LOCAL_ICDAR_DIR / "final_eval.json") as f:
        icdar_gt = json.load(f)

    annotations = [
        {"filename": fname, "gt_html": obj["html"], "type": obj["type"]}
        for fname, obj in icdar_gt.items()
    ]
    if N_TABLES is not None:
        annotations = annotations[:N_TABLES]

    print(f"Total tables: {len(annotations)}  Workers: {N_WORKERS}")

    # Split into even chunks
    chunk_size = math.ceil(len(annotations) / N_WORKERS)
    chunks     = [annotations[i:i + chunk_size] for i in range(0, len(annotations), chunk_size)]
    worker_ids = list(range(len(chunks)))

    print(f"Chunk size: {chunk_size}  Actual workers: {len(chunks)}")

    # Fan out — all chunks run in parallel
    all_results = []
    for chunk_results in process_chunk.starmap(zip(chunks, worker_ids)):
        all_results.extend(chunk_results)

    # Merge and save locally
    run_id   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = f"teds_labels_{run_id}.csv"
    df       = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)

    print(f"\nDone — {len(all_results)} processed")
    print(f"Saved locally to: {out_path}")

    clean = df[df["label"] != -1]
    print(f"\nTEDS stats:\n{df['teds_score'].describe()}")
    print(f"\nLabel distribution:\n{clean['label'].value_counts()}")
