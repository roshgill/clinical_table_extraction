"""
UniTable Binary Classifier — Modal Training Script
===================================================
Trains a lightweight classification head on top of UniTable's frozen ViT encoder
on an A100 GPU via Modal. Mirrors train_classifier.py but runs in the cloud.

Usage:
    modal run scripts/train_classifier_modal.py

Pull results after completion:
    modal volume get pubtabnet-data /checkpoints/best_classifier.pt ./checkpoints/best_classifier.pt
    modal volume get pubtabnet-data /checkpoints/threshold.txt ./checkpoints/threshold.txt
    modal volume get pubtabnet-data /checkpoints/pr_curve.png ./checkpoints/pr_curve.png
"""

import modal
from pathlib import Path

# ── Local paths ───────────────────────────────────────────────────────────────
ROOT                  = Path(__file__).parent.parent
LOCAL_ICDAR_IMAGES    = ROOT / "Training set for encoder binary classifier pipeline" / "icdar-task-b" / "final_eval"
LOCAL_PUBTAB_IMAGES   = ROOT / "pubtabnet" / "images"
LOCAL_WEIGHTS         = ROOT / "Huma-Huma" / "unitable" / "experiments" / "unitable_weights"
LOCAL_VOCAB           = ROOT / "Huma-Huma" / "unitable" / "vocab"
LOCAL_ICDAR_CSV       = ROOT / "teds_labels_20260410_030718.csv"
LOCAL_PUBTAB_CSV      = ROOT / "teds_labels_pubtabnet_full_20260411_200347.csv"

# ── Remote paths ──────────────────────────────────────────────────────────────
REMOTE_ICDAR_IMAGES   = "/images/icdar"
REMOTE_PUBTAB_IMAGES  = "/images/pubtabnet"
REMOTE_WEIGHTS        = "/weights"
REMOTE_VOCAB          = "/vocab"
REMOTE_LABELS_CSV     = "/labels/teds_labels_combined.csv"
REMOTE_CKPT_DIR       = "/checkpoints"

# ── Volume for outputs ────────────────────────────────────────────────────────
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
        "scikit-learn",
        "pandas",
        "Pillow",
        "matplotlib",
    ])
    .run_commands([
        "git clone https://github.com/poloclub/unitable.git /unitable",
        "cd /unitable && pip install -e . --quiet",
    ])
    .add_local_dir(str(LOCAL_ICDAR_IMAGES),  remote_path=REMOTE_ICDAR_IMAGES)
    .add_local_dir(str(LOCAL_PUBTAB_IMAGES), remote_path=REMOTE_PUBTAB_IMAGES)
    .add_local_dir(str(LOCAL_WEIGHTS),       remote_path=REMOTE_WEIGHTS)
    .add_local_dir(str(LOCAL_VOCAB),         remote_path=REMOTE_VOCAB)
    .add_local_file(str(LOCAL_ICDAR_CSV),    remote_path="/labels/icdar.csv")
    .add_local_file(str(LOCAL_PUBTAB_CSV),   remote_path="/labels/pubtabnet.csv")
)

app = modal.App("unitable-classifier-training", image=image)

# ── Training config ───────────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 32
LR         = 3e-5
EPOCHS     = 30
PATIENCE   = 5
DROPOUT    = 0.3
D_MODEL    = 768
POOL_DIM   = D_MODEL * 3   # concat [mean, max, std]
PATCH_SIZE = 16
NHEAD      = 12


@app.function(
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=7200,   # 2 hours
    memory=32768,
)
def train():
    import sys, os, math
    from pathlib import Path
    from functools import partial

    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        precision_recall_curve, confusion_matrix,
        classification_report, roc_auc_score,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import tokenizers as tk

    sys.path.insert(0, "/unitable")
    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    MEAN = [0.86597056, 0.88463002, 0.87491087]
    STD  = [0.20686628, 0.18201602, 0.18485524]

    torch.manual_seed(SEED)
    os.makedirs(REMOTE_CKPT_DIR, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    def pad_to_square(image: Image.Image, fill: int = 255) -> Image.Image:
        w, h    = image.size
        max_dim = max(w, h)
        padded  = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
        padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        return padded

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    class TableDataset(Dataset):
        """Loads images from ICDAR or PubTabNet based on the 'source' column.
        ICDAR rows have no source column — defaults to icdar.
        """
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row    = self.df.iloc[idx]
            source = row.get("source", "icdar") if isinstance(row.get("source", "icdar"), str) else "icdar"
            if source == "pubtabnet":
                img_dir = REMOTE_PUBTAB_IMAGES
            else:
                img_dir = REMOTE_ICDAR_IMAGES
            path  = os.path.join(img_dir, row["filename"])
            image = Image.open(path).convert("RGB")
            image = pad_to_square(image)
            tensor = transform(image)
            label  = torch.tensor(float(row["label"]), dtype=torch.float32)
            return tensor, label

    # ── Model ──────────────────────────────────────────────────────────────────
    class UniTableClassifier(nn.Module):
        def __init__(self):
            super().__init__()

            backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)
            encoder  = Encoder(
                d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
                activation="gelu", norm_first=True, nlayer=12, ff_ratio=4,
            )
            decoder  = Decoder(
                d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
                activation="gelu", norm_first=True, nlayer=4, ff_ratio=4,
            )
            vocab_path = f"{REMOTE_VOCAB}/vocab_html.json"
            vocab      = tk.Tokenizer.from_file(vocab_path)

            full_model = EncoderDecoder(
                backbone=backbone, encoder=encoder, decoder=decoder,
                vocab_size=vocab.get_vocab_size(), d_model=D_MODEL,
                padding_idx=vocab.token_to_id("<pad>"),
                max_seq_len=784, dropout=0.2,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            weights_path = f"{REMOTE_WEIGHTS}/unitable_large_structure.pt"
            full_model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )

            self.backbone = full_model.backbone
            self.encoder  = full_model.encoder
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False

            # Classification head — LayerNorm tames the large encoder output scale
            # before the first Linear (encoder outputs routinely hit std=6, max=125)
            self.head = nn.Sequential(
                nn.LayerNorm(POOL_DIM),
                nn.Dropout(DROPOUT),
                nn.Linear(POOL_DIM, 256),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(256, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            patches   = self.backbone(x)                               # (batch, seq_len, 768)
            memory    = self.encoder(patches)                          # (batch, seq_len, 768)
            mean_pool = memory.mean(dim=1)                             # (batch, 768)
            max_pool  = memory.max(dim=1).values                       # (batch, 768)
            std_pool  = memory.std(dim=1) + 1e-6                       # (batch, 768)
            pooled    = torch.cat([mean_pool, max_pool, std_pool], dim=1)  # (batch, 2304)
            return self.head(pooled).squeeze(-1)                       # (batch,) — raw logits

    # ── Data — merge ICDAR + PubTabNet ────────────────────────────────────────
    df_icdar  = pd.read_csv("/labels/icdar.csv")
    df_icdar["source"] = "icdar"
    df_pubtab = pd.read_csv("/labels/pubtabnet.csv")
    # pubtabnet CSV already has source column

    df       = pd.concat([df_icdar, df_pubtab], ignore_index=True)
    df_clean = df[df["label"].isin([0, 1]) & (df["teds_score"] >= 0)].copy()

    print(f"Combined dataset: {len(df_clean)} rows")
    print(f"  ICDAR:     {(df_clean['source']=='icdar').sum()}")
    print(f"  PubTabNet: {(df_clean['source']=='pubtabnet').sum()}")
    print(f"  Positive (1): {(df_clean['label']==1).sum()}")
    print(f"  Negative (0): {(df_clean['label']==0).sum()}")

    df_train, df_temp = train_test_split(df_clean, test_size=0.30, stratify=df_clean["label"], random_state=SEED)
    df_val,   df_test = train_test_split(df_temp,  test_size=0.50, stratify=df_temp["label"],  random_state=SEED)
    print(f"Splits — train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")

    train_ds = TableDataset(df_train)
    val_ds   = TableDataset(df_val)
    test_ds  = TableDataset(df_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model = UniTableClassifier().to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    pos_weight = torch.tensor(
        [len(df_clean[df_clean["label"]==0]) / len(df_clean[df_clean["label"]==1])],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds       = (torch.sigmoid(logits) >= 0.5).float()
            correct    += (preds == labels).sum().item()
            total      += len(labels)

        train_loss /= total
        train_acc   = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_dl:
                images, labels = images.to(device), labels.to(device)
                logits  = model(images)
                loss    = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds    = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total   += len(labels)

        val_loss /= total
        val_acc   = correct / total
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
            }, f"{REMOTE_CKPT_DIR}/best_classifier.pt")
            volume.commit()
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Threshold tuning on val set ────────────────────────────────────────────
    print("\nTuning decision threshold on val set...")
    ckpt = torch.load(f"{REMOTE_CKPT_DIR}/best_classifier.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_dl:
            logits = model(images.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores  = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx   = f1_scores.argmax()
    best_thresh = thresholds[best_idx]

    print(f"Optimal threshold: {best_thresh:.4f}  "
          f"(precision={precision[best_idx]:.3f}  recall={recall[best_idx]:.3f}  "
          f"F1={f1_scores[best_idx]:.3f})")

    with open(f"{REMOTE_CKPT_DIR}/threshold.txt", "w") as f:
        f.write(str(best_thresh))

    # ── Final evaluation on test set ───────────────────────────────────────────
    print("\nEvaluating on test set...")
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_dl:
            logits = model(images.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds      = (all_probs >= best_thresh).astype(int)

    print(classification_report(all_labels, preds, target_names=["negative", "positive"]))
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(all_labels, preds)}")

    # Precision-recall curve plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="steelblue")
    ax.axvline(recall[best_idx], color="red", linestyle="--", label=f"threshold={best_thresh:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (val set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REMOTE_CKPT_DIR}/pr_curve.png", dpi=150)
    plt.close()

    volume.commit()
    print(f"\nDone. Artifacts saved to volume under {REMOTE_CKPT_DIR}/")
    print("  best_classifier.pt")
    print("  threshold.txt")
    print("  pr_curve.png")


@app.function(
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=300,
)
def diagnose():
    import sys, os
    from functools import partial
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torchvision import transforms
    import tokenizers as tk

    sys.path.insert(0, "/unitable")
    from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n{'='*60}")

    # ── Build model (same as train()) ──────────────────────────────────────────
    backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)
    encoder  = Encoder(
        d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
        activation="gelu", norm_first=True, nlayer=12, ff_ratio=4,
    )
    decoder  = Decoder(
        d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
        activation="gelu", norm_first=True, nlayer=4, ff_ratio=4,
    )
    vocab      = tk.Tokenizer.from_file(f"{REMOTE_VOCAB}/vocab_html.json")
    full_model = EncoderDecoder(
        backbone=backbone, encoder=encoder, decoder=decoder,
        vocab_size=vocab.get_vocab_size(), d_model=D_MODEL,
        padding_idx=vocab.token_to_id("<pad>"),
        max_seq_len=784, dropout=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    full_model.load_state_dict(
        torch.load(f"{REMOTE_WEIGHTS}/unitable_large_structure.pt",
                   map_location="cpu", weights_only=True)
    )

    enc_backbone = full_model.backbone.to(device)
    enc_encoder  = full_model.encoder.to(device)
    for p in enc_backbone.parameters(): p.requires_grad = False
    for p in enc_encoder.parameters():  p.requires_grad = False

    # ── Diagnostic 1: Verify frozen ────────────────────────────────────────────
    print("DIAGNOSTIC 1 — Are weights frozen?")
    enc_frozen  = all(not p.requires_grad for p in enc_encoder.parameters())
    back_frozen = all(not p.requires_grad for p in enc_backbone.parameters())
    print(f"  Encoder frozen:  {enc_frozen}")
    print(f"  Backbone frozen: {back_frozen}")

    # ── Load one real batch from ICDAR ─────────────────────────────────────────
    MEAN = [0.86597056, 0.88463002, 0.87491087]
    STD  = [0.20686628, 0.18201602, 0.18485524]
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def pad_to_square(image, fill=255):
        w, h = image.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (fill, fill, fill))
        padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
        return padded

    df    = pd.read_csv(REMOTE_LABELS_CSV)
    files = df[df["label"].isin([0, 1])]["filename"].tolist()[:8]
    imgs  = []
    for fname in files:
        img = Image.open(os.path.join(REMOTE_IMAGES_DIR, fname)).convert("RGB")
        imgs.append(transform(pad_to_square(img)))
    batch = torch.stack(imgs).to(device)
    print(f"\n  Sample batch: {batch.shape}  (dtype={batch.dtype})")
    print(f"  Batch pixel  min={batch.min():.3f}  max={batch.max():.3f}  "
          f"mean={batch.mean():.3f}  NaN={torch.isnan(batch).any().item()}")

    # ── Diagnostic 2: Inspect encoder output ──────────────────────────────────
    print("\nDIAGNOSTIC 2 — Encoder output stats")
    with torch.no_grad():
        patches   = enc_backbone(batch)
        memory    = enc_encoder(patches)
        mean_pool = memory.mean(dim=1)
        max_pool  = memory.max(dim=1).values
        std_pool  = memory.std(dim=1) + 1e-6

    print(f"  patches shape:   {patches.shape}")
    print(f"  memory shape:    {memory.shape}")
    print(f"  mean_pool  — mean={mean_pool.mean():.4f}  std={mean_pool.std():.4f}  "
          f"min={mean_pool.min():.4f}  max={mean_pool.max():.4f}  "
          f"NaN={torch.isnan(mean_pool).any().item()}")
    print(f"  max_pool   — mean={max_pool.mean():.4f}  std={max_pool.std():.4f}  "
          f"NaN={torch.isnan(max_pool).any().item()}")
    print(f"  std_pool   — mean={std_pool.mean():.4f}  std={std_pool.std():.4f}  "
          f"NaN={torch.isnan(std_pool).any().item()}")

    pooled = torch.cat([mean_pool, max_pool, std_pool], dim=1)
    print(f"  pooled shape:    {pooled.shape}  "
          f"NaN={torch.isnan(pooled).any().item()}")

    # ── Diagnostic 3: Can the head learn on random encoder output? ─────────────
    print("\nDIAGNOSTIC 3 — Can head learn with random (untrained) encoder?")
    rand_backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE).to(device)
    rand_encoder  = Encoder(
        d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
        activation="gelu", norm_first=True, nlayer=12, ff_ratio=4,
    ).to(device)
    # both random-initialized — do NOT freeze
    head = nn.Sequential(
        nn.LayerNorm(POOL_DIM),
        nn.Dropout(DROPOUT), nn.Linear(POOL_DIM, 256),
        nn.ReLU(),
        nn.Dropout(DROPOUT), nn.Linear(256, 1),
    ).to(device)
    opt  = torch.optim.AdamW(
        list(rand_backbone.parameters()) +
        list(rand_encoder.parameters()) +
        list(head.parameters()), lr=1e-3
    )
    crit = nn.BCEWithLogitsLoss()
    labels = torch.zeros(batch.shape[0], device=device)
    labels[:4] = 1.0

    losses = []
    for step in range(20):
        opt.zero_grad()
        with torch.set_grad_enabled(True):
            p  = rand_backbone(batch)
            m  = rand_encoder(p)
            mp = m.mean(dim=1)
            xp = m.max(dim=1).values
            sp = m.std(dim=1) + 1e-6
            po = torch.cat([mp, xp, sp], dim=1)
            lo = crit(head(po).squeeze(-1), labels)
        lo.backward()
        opt.step()
        losses.append(lo.item())

    print(f"  Loss over 20 steps (random encoder, unfrozen):")
    print(f"    step 0:  {losses[0]:.4f}")
    print(f"    step 9:  {losses[9]:.4f}")
    print(f"    step 19: {losses[19]:.4f}")
    decreasing = losses[19] < losses[0]
    print(f"  Loss decreasing: {decreasing}  "
          f"({'head + encoder can learn' if decreasing else 'TRAINING LOOP BUG'})")

    print("\n" + "="*60)
    print("Diagnostics complete.")


@app.local_entrypoint()
def main():
    train.remote()


@app.local_entrypoint()
def run_diagnose():
    diagnose.remote()
