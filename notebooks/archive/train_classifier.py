"""
UniTable Binary Classifier — Training Script
============================================
Trains a lightweight classification head on top of UniTable's frozen ViT encoder.
Labels: 1 = UniTable handles well (TEDS >= 0.90), 0 = UniTable struggles (TEDS <= 0.85)

Usage:
    python scripts/train_classifier.py

Outputs:
    checkpoints/best_classifier.pt  — best val loss checkpoint
    checkpoints/threshold.txt       — optimal decision threshold from val set
"""

import sys, os, re
import math
import json
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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
IMAGES_DIR   = ROOT / "Training set for encoder binary classifier pipeline" / "icdar-task-b" / "final_eval"
LABELS_CSV   = ROOT / "teds_labels_20260410_030718.csv"
WEIGHTS_PATH = ROOT / "Huma-Huma" / "unitable" / "experiments" / "unitable_weights" / "unitable_large_structure.pt"
UNITABLE_SRC = ROOT / "Huma-Huma" / "unitable"
CKPT_DIR     = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(UNITABLE_SRC))
from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
BATCH_SIZE  = 32
LR          = 3e-5
EPOCHS      = 30
PATIENCE    = 5       # early stopping
DROPOUT     = 0.3
D_MODEL     = 768
POOL_DIM    = D_MODEL * 3   # concat [mean, max, std]
PATCH_SIZE  = 16
NHEAD       = 12

MEAN = [0.86597056, 0.88463002, 0.87491087]
STD  = [0.20686628, 0.18201602, 0.18485524]

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


# ── Dataset ───────────────────────────────────────────────────────────────────
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


class ICDARDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: Path):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = self.images_dir / row["filename"]
        image = Image.open(path).convert("RGB")
        image = pad_to_square(image)
        tensor = transform(image)
        label  = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


# ── Model ─────────────────────────────────────────────────────────────────────
class UniTableClassifier(nn.Module):
    def __init__(self, weights_path: Path):
        super().__init__()

        # Build UniTable encoder architecture (structure model)
        backbone = ImgLinearBackbone(d_model=D_MODEL, patch_size=PATCH_SIZE)
        encoder  = Encoder(
            d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
            activation="gelu", norm_first=True, nlayer=12, ff_ratio=4,
        )
        decoder  = Decoder(
            d_model=D_MODEL, nhead=NHEAD, dropout=0.2,
            activation="gelu", norm_first=True, nlayer=4, ff_ratio=4,
        )
        import tokenizers as tk
        vocab_path = UNITABLE_SRC / "vocab" / "vocab_html.json"
        vocab      = tk.Tokenizer.from_file(str(vocab_path))

        full_model = EncoderDecoder(
            backbone=backbone, encoder=encoder, decoder=decoder,
            vocab_size=vocab.get_vocab_size(), d_model=D_MODEL,
            padding_idx=vocab.token_to_id("<pad>"),
            max_seq_len=784, dropout=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        full_model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )

        # Keep only backbone + encoder, freeze them
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
        # x: (batch, 3, 448, 448)
        patches   = self.backbone(x)                              # (batch, seq_len, 768)
        memory    = self.encoder(patches)                         # (batch, seq_len, 768)
        mean_pool = memory.mean(dim=1)                            # (batch, 768)
        max_pool  = memory.max(dim=1).values                      # (batch, 768)
        std_pool  = memory.std(dim=1) + 1e-6                        # (batch, 768)
        pooled    = torch.cat([mean_pool, max_pool, std_pool], dim=1)  # (batch, 2304)
        return self.head(pooled).squeeze(-1)                      # (batch,) — raw logits


# ── Data preparation ──────────────────────────────────────────────────────────
df = pd.read_csv(LABELS_CSV)

# Keep only clean labels (0 and 1), drop ambiguous and negative TEDS
df_clean = df[df["label"].isin([0, 1]) & (df["teds_score"] >= 0)].copy()
print(f"Clean dataset: {len(df_clean)} rows")
print(f"  Positive (1): {(df_clean['label']==1).sum()}")
print(f"  Negative (0): {(df_clean['label']==0).sum()}")

# 70/15/15 stratified split
df_train, df_temp = train_test_split(df_clean, test_size=0.30, stratify=df_clean["label"], random_state=SEED)
df_val,   df_test = train_test_split(df_temp,  test_size=0.50, stratify=df_temp["label"],  random_state=SEED)

print(f"\nSplits — train: {len(df_train)}  val: {len(df_val)}  test: {len(df_test)}")

train_ds = ICDARDataset(df_train, IMAGES_DIR)
val_ds   = ICDARDataset(df_val,   IMAGES_DIR)
test_ds  = ICDARDataset(df_test,  IMAGES_DIR)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ── Model, loss, optimizer ────────────────────────────────────────────────────
model = UniTableClassifier(WEIGHTS_PATH).to(device)

# Only train the head
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"\nTrainable params: {sum(p.numel() for p in trainable):,}")

pos_weight = torch.tensor([len(df_clean[df_clean["label"]==0]) /
                            len(df_clean[df_clean["label"]==1])], device=device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── Training loop ─────────────────────────────────────────────────────────────
if WANDB:
    try:
        wandb.init(project="huma-table-classifier", config={
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "dropout": DROPOUT, "patience": PATIENCE,
            "pos_weight": pos_weight.item(),
            "train_size": len(df_train), "val_size": len(df_val), "test_size": len(df_test),
        })
    except Exception:
        WANDB = False
        print("W&B unavailable — logging to stdout only")

best_val_loss = float("inf")
patience_ctr  = 0

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────
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

    # ── Validate ───────────────────────────────────────────────────────────
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

    if WANDB: wandb.log({"train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch})

    # ── Checkpoint ─────────────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save({
            "epoch":      epoch,
            "model_state": model.state_dict(),
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        }, CKPT_DIR / "best_classifier.pt")
        print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break


# ── Threshold tuning on val set ───────────────────────────────────────────────
print("\nTuning decision threshold on val set...")
model.load_state_dict(torch.load(CKPT_DIR / "best_classifier.pt")["model_state"])
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

with open(CKPT_DIR / "threshold.txt", "w") as f:
    f.write(str(best_thresh))

if WANDB: wandb.log({"best_threshold": best_thresh, "best_f1": f1_scores[best_idx]})


# ── Final evaluation on test set ──────────────────────────────────────────────
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
plt.savefig(CKPT_DIR / "pr_curve.png", dpi=150)
if WANDB:
    wandb.log({"pr_curve": wandb.Image(str(CKPT_DIR / "pr_curve.png"))})
    wandb.finish()
print(f"\nDone. Checkpoint: {CKPT_DIR / 'best_classifier.pt'}")
