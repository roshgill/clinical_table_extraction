"""
PubTabNet Download Script
=========================
Downloads 25,600 samples from ajimeno/PubTabNet using your HuggingFace API key.
Saves images + annotation JSONL locally so they can be used for TEDS labeling.

First run prints the full schema so we can verify annotation fields are present.

Usage:
    python scripts/download_pubtabnet.py

Outputs:
    pubtabnet/images/<key>.png
    pubtabnet/annotations.jsonl   — one line per image: {key, gt_html}
"""

import os, io, sys, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN  = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
ROOT      = Path(__file__).parent.parent
OUT_DIR   = ROOT / "pubtabnet"
IMG_DIR   = OUT_DIR / "images"
ANNO_FILE = OUT_DIR / "annotations.jsonl"
N_SAMPLES = 25_600

IMG_DIR.mkdir(parents=True, exist_ok=True)

if not HF_TOKEN:
    print("No HF_TOKEN found in .env — add HF_TOKEN=hf_... to .env and retry.")
    sys.exit(1)

print(f"Output directory : {OUT_DIR}")
print(f"Target samples   : {N_SAMPLES}")

# ── Check existing progress ────────────────────────────────────────────────────
existing = list(IMG_DIR.glob("*.png"))
print(f"Already downloaded: {len(existing)} images")
if len(existing) >= N_SAMPLES:
    print("Already complete — nothing to do.")
    sys.exit(0)

# ── Load dataset ───────────────────────────────────────────────────────────────
from datasets import load_dataset
from PIL import Image

print("\nLoading one row (non-streaming) to inspect full schema...")
ds_peek = load_dataset(
    "ajimeno/PubTabNet",
    split="train[:1]",
    token=HF_TOKEN,
)
first = ds_peek[0]
print("\nAvailable fields:")
for k, v in first.items():
    vtype = type(v).__name__
    if isinstance(v, dict):
        preview = f"dict keys: {list(v.keys())}"
    elif isinstance(v, list):
        preview = f"list[{len(v)}]  first item type: {type(v[0]).__name__ if v else 'empty'}"
    elif isinstance(v, (str, bytes)) and len(str(v)) > 80:
        preview = str(v)[:80] + "..."
    else:
        preview = repr(v)
    print(f"  {k!r:25s} ({vtype}): {preview}")

HAS_HTML = "html" in first
print(f"\nAnnotation field 'html' present: {HAS_HTML}")
if not HAS_HTML:
    print("No 'html' field found — images only, cannot compute TEDS without annotations.")
    print("Saving images anyway so we have them ready.")

# ── Stream and save ────────────────────────────────────────────────────────────
print(f"\nStreaming {N_SAMPLES} samples...")
ds = load_dataset(
    "ajimeno/PubTabNet",
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

# Build set of already-downloaded keys to resume interrupted runs
done_keys    = {p.stem for p in existing}
n_already    = len(done_keys)
n_remaining  = N_SAMPLES - n_already
print(f"Need {n_remaining} more images")

saved   = 0
skipped = 0
anno_lines = []

# Load existing annotations if resuming
if ANNO_FILE.exists() and HAS_HTML:
    with open(ANNO_FILE) as f:
        anno_lines = [line.strip() for line in f if line.strip()]
    print(f"Resuming — {len(anno_lines)} annotations already saved")

from tqdm import tqdm

for row in tqdm(ds.take(N_SAMPLES + 1000), desc="Downloading", unit="img"):
    if saved >= n_remaining:   # compare only new saves against remaining target
        break

    try:
        # Image field is already a PIL Image when loaded via Parquet/streaming
        raw = row.get("png") or row.get("image")
        if raw is None:
            skipped += 1
            continue

        key = row["__key__"].replace("/", "_")   # flatten path separators

        # Skip already-downloaded
        if key in done_keys:
            continue

        # Save image
        if isinstance(raw, bytes):
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        else:
            img = raw.convert("RGB")   # already PIL

        img.save(IMG_DIR / f"{key}.png")
        done_keys.add(key)
        saved += 1

        # Save annotation if available
        if HAS_HTML:
            html_data = row["html"]
            anno_lines.append(json.dumps({
                "key":      key,
                "html_raw": html_data,   # keep raw dict — GT HTML reconstructed at labeling time
            }))

    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"\n  Skipped ({e})")
        continue

    # Checkpoint every 500
    if saved % 500 == 0 and saved > 0:
        if HAS_HTML and anno_lines:
            with open(ANNO_FILE, "w") as f:
                f.write("\n".join(anno_lines) + "\n")
        print(f"  {saved + len(existing)} saved  {skipped} skipped")

# ── Final save ────────────────────────────────────────────────────────────────
if HAS_HTML and anno_lines:
    with open(ANNO_FILE, "w") as f:
        f.write("\n".join(anno_lines) + "\n")

total = saved + len(existing)
print(f"\nDone — {total} images in {IMG_DIR}")

# ── Build annotations.jsonl from PubTabNet_2.0.0.jsonl ────────────────────────
# The annotation JSONL was extracted from the TAR separately.
# Join on filename stem → emit one line per downloaded image.
RAW_JSONL = OUT_DIR / "PubTabNet_2.0.0.jsonl"
if RAW_JSONL.exists():
    print(f"\nBuilding {ANNO_FILE} from {RAW_JSONL.name}...")
    downloaded_stems = {p.stem for p in IMG_DIR.glob("*.png")}
    written = 0
    with open(RAW_JSONL) as fin, open(ANNO_FILE, "w") as fout:
        for line in fin:
            row = json.loads(line)
            stem = Path(row["filename"]).stem   # e.g. PMC1797029_008_00
            # __key__ flattens slashes: pubtabnet_test_PMC1797029_008_00
            flat_key = f"pubtabnet_test_{stem}" if "test" in row.get("split","") \
                  else f"pubtabnet_train_{stem}"
            if flat_key in downloaded_stems or stem in downloaded_stems:
                fout.write(json.dumps({
                    "key":      flat_key,
                    "filename": row["filename"],
                    "split":    row.get("split", ""),
                    "html_raw": row["html"],
                }) + "\n")
                written += 1
    print(f"Wrote {written} annotation entries → {ANNO_FILE}")
else:
    print(f"\nAnnotation JSONL not found at {RAW_JSONL}")
    print("Extract it with:")
    print("  tar -xzf ~/.cache/huggingface/hub/datasets--ajimeno--PubTabNet/snapshots/*/pubtabnet.tar.gz \\")
    print("      -C pubtabnet --strip-components=1 pubtabnet/PubTabNet_2.0.0.jsonl")
