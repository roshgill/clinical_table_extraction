"""
PubTabNet Local TEDS Test
=========================
Runs UniTable on N PubTabNet tables locally and compares against GT HTML.
Use this to verify the GT reconstruction and TEDS pipeline before
kicking off the full Modal run.

Usage:
    python scripts/test_pubtabnet_local.py          # 5 tables
    python scripts/test_pubtabnet_local.py --n 20   # 20 tables
"""

import sys, os, re, json, argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import tokenizers as tk
from PIL import Image
from torchvision import transforms

ROOT          = Path(__file__).parent.parent
UNITABLE_SRC  = ROOT / "Huma-Huma" / "unitable"
WEIGHTS_DIR   = UNITABLE_SRC / "experiments" / "unitable_weights"
VOCAB_DIR     = UNITABLE_SRC / "vocab"
IMAGES_DIR    = ROOT / "pubtabnet" / "images"
ANNO_FILE     = ROOT / "pubtabnet" / "annotations.jsonl"

sys.path.insert(0, str(UNITABLE_SRC))
from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder  # type: ignore
from src.utils import (  # type: ignore
    subsequent_mask, pred_token_within_range, greedy_sampling,
    bbox_str_to_token_list, cell_str_to_token_list,
    html_str_to_token_list, build_table_from_html_and_cell,
    html_table_template,
)
from src.vocab import HTML_TOKENS, TASK_TOKENS, RESERVED_TOKENS, BBOX_TOKENS  # type: ignore
from src.utils.teds import TEDS  # type: ignore

VALID_HTML_TOKEN   = ["<eos>"] + HTML_TOKENS
VALID_BBOX_TOKEN   = ["<eos>"] + BBOX_TOKENS
INVALID_CELL_TOKEN = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}\n")

MEAN = [0.86597056, 0.88463002, 0.87491087]
STD  = [0.20686628, 0.18201602, 0.18485524]
d_model, patch_size, nhead, dropout = 768, 16, 12, 0.2


# ── Model loading ──────────────────────────────────────────────────────────────

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
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return vocab, model.to(device).eval()

print("Loading UniTable models...")
vocab_html, model_html = make_model(VOCAB_DIR / "vocab_html.json",    784,  WEIGHTS_DIR / "unitable_large_structure.pt")
vocab_bbox, model_bbox = make_model(VOCAB_DIR / "vocab_bbox.json",    1024, WEIGHTS_DIR / "unitable_large_bbox.pt")
vocab_cell, model_cell = make_model(VOCAB_DIR / "vocab_cell_6k.json", 200,  WEIGHTS_DIR / "unitable_large_content.pt")
print("Models loaded.\n")


# ── Inference helpers ──────────────────────────────────────────────────────────

def pad_to_square(image: Image.Image, fill: int = 255) -> Image.Image:
    w, h    = image.size
    max_dim = max(w, h)
    padded  = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    return padded

def to_tensor(image: Image.Image, size=(448, 448)):
    T = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return T(image).to(device).unsqueeze(0)  # type: ignore[union-attr]

def decode(model, image, prefix, max_len, eos_id, whitelist=None, blacklist=None):
    with torch.no_grad():
        memory  = model.encode(image)
        context = torch.tensor(prefix, dtype=torch.int32).repeat(image.shape[0], 1).to(device)
    for _ in range(max_len):
        if all(eos_id in k for k in context):
            break
        with torch.no_grad():
            mask   = subsequent_mask(context.shape[1]).to(device)
            logits = model.decode(memory, context, tgt_mask=mask, tgt_padding_mask=None)
            logits = model.generator(logits)[:, -1, :]
        logits = pred_token_within_range(logits.detach(), white_list=whitelist, black_list=blacklist)
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
    img_t    = to_tensor(padded, (448, 448))

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
        return None, None

    pb_orig = [
        [max(0, int(b[0]*scale)-pad_x), max(0, int(b[1]*scale)-pad_y),
         min(orig_w, int(b[2]*scale)-pad_x), min(orig_h, int(b[3]*scale)-pad_y)]
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

    pred_html = html_table_template("".join(build_table_from_html_and_cell(ph, pc)))
    return pred_html, ph


# ── GT reconstruction ──────────────────────────────────────────────────────────

def reconstruct_gt_html(html_raw: dict) -> str:
    """
    PubTabNet stores structure as separate '<td>' / '</td>' tokens.
    colspan/rowspan tags span multiple tokens: ['<td', ' colspan="2"', '>'].
    Walk the token list, collect complete opening tags, then inject cell content.
    Cell tokens are character-level — join with '' not ' '.
    """
    structure_tokens = html_raw["structure"]["tokens"]
    cells            = html_raw["cells"]
    cell_texts       = ["".join(c["tokens"]) for c in cells]

    result   = []
    cell_idx = 0
    i        = 0
    while i < len(structure_tokens):
        tok = structure_tokens[i]
        if tok.startswith("<td"):
            full_tag = tok
            while not full_tag.endswith(">"):
                i += 1
                full_tag += structure_tokens[i]
            result.append(full_tag)
            if cell_idx < len(cell_texts):
                result.append(cell_texts[cell_idx])
                cell_idx += 1
        else:
            result.append(tok)
        i += 1

    return html_table_template("".join(result))


# ── Main ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=5, help="Number of tables to test")
args = parser.parse_args()

teds_metric = TEDS(structure_only=False, ignore_nodes="b")

with open(ANNO_FILE) as f:
    annotations = [json.loads(line) for line in f if line.strip()]

print(f"Testing {args.n} tables from {len(annotations)} annotated\n")
print("=" * 70)

scores = []
for anno in annotations[:args.n]:
    key      = anno["key"]
    img_path = IMAGES_DIR / f"{key}.png"

    if not img_path.exists():
        print(f"[SKIP] Image not found: {img_path.name}")
        continue

    print(f"Table: {key}")
    image = Image.open(img_path).convert("RGB")
    print(f"  Image size: {image.size}")

    # Ground truth
    gt_html = reconstruct_gt_html(anno["html_raw"])
    n_cells = len(anno["html_raw"]["cells"])
    n_slots = sum(1 for t in anno["html_raw"]["structure"]["tokens"] if t.startswith("<td"))
    print(f"  GT cells: {n_cells}  structure slots: {n_slots}")

    # Prediction
    pred_html, structure_tokens = run_unitable(image)
    if pred_html is None:
        print("  RESULT: UniTable returned None (bbox decoder found no cells)")
        print()
        continue

    # TEDS
    score = teds_metric.evaluate(pred_html, gt_html)
    label = 1 if score >= 0.95 else 0 if score <= 0.85 else -1
    scores.append(score)

    print(f"  TEDS: {score:.4f}  →  label={label}")
    print()
    print("  GT HTML (first 300 chars):")
    print("  " + gt_html[gt_html.find("<thead>"):gt_html.find("<thead>")+300].replace("\n", " "))
    print()
    print("  PRED HTML (first 300 chars):")
    print("  " + pred_html[pred_html.find("<thead>"):pred_html.find("<thead>")+300].replace("\n", " "))
    print()
    print("-" * 70)

if scores:
    import statistics
    print(f"\nSummary ({len(scores)} tables):")
    print(f"  Mean TEDS : {statistics.mean(scores):.4f}")
    print(f"  Min  TEDS : {min(scores):.4f}")
    print(f"  Max  TEDS : {max(scores):.4f}")
    print(f"  Label=1 (≥0.95): {sum(1 for s in scores if s >= 0.95)}")
    print(f"  Label=0 (≤0.85): {sum(1 for s in scores if s <= 0.85)}")
    print(f"  Label=-1 (ambig): {sum(1 for s in scores if 0.85 < s < 0.95)}")
