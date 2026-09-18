"""
train_finetune.py -- Stage 1 (v4 Section 4). Run this in Colab, on GPU.

CRITICAL: this imports model_api.py directly rather than redefining the
architecture/preprocessing -- that's the whole point of v4 decision 2
(shared module, no train/inference skew). Upload model_api.py to your Colab
session alongside this file, or !pip install/copy it into the working dir.

Output: rdamage_mobilenetv2.pt -- upload this to HDFS at
hdfs:///models/rdamage_mobilenetv2.pt when done. That path is what
spark_pipeline.py expects.

Expected input: a directory of labeled 224x224 patches, organized as
    patch_dataset/
        train/
            background/*.jpg
            D00/*.jpg
            D10/*.jpg
            D20/*.jpg
            D40/*.jpg
        val/
            (same structure)
This is what Person 2/3's Stage 0 (Spark patch dataset build) should produce
and hand to you -- confirm the exact folder layout with them before running
this, since this script assumes ImageFolder-compatible structure.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_api import build_model, preprocess, CLASSES, NUM_CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Config (v4 decision 1: lr ~1e-4, 10-20 epochs, early stopping) ----
TRAIN_DIR = "patch_dataset/train"
VAL_DIR = "patch_dataset/val"
LR = 1e-4
MAX_EPOCHS = 20
PATIENCE = 4  # early stopping: stop if val F1 doesn't improve for this many epochs
BATCH_SIZE = 64
OUTPUT_PATH = "rdamage_mobilenetv2.pt"


def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    """
    Class-weighted loss (v4 decision 1 and 5). ImageFolder's classes are
    sorted alphabetically -- confirm this matches model_api.CLASSES order,
    or the weights will be misapplied to the wrong classes.
    """
    from collections import Counter
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    n_classes = len(dataset.classes)
    weights = torch.tensor(
        [total / (n_classes * counts.get(i, 1)) for i in range(n_classes)],
        dtype=torch.float32,
    )
    return weights


def evaluate(model, loader) -> dict:
    """Patch-level accuracy + weighted F1 on the validation set."""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(labels.numpy())
    return {
        "accuracy": accuracy_score(trues, preds),
        "weighted_f1": f1_score(trues, preds, average="weighted"),
    }


def train():
    print(f"[train] device = {DEVICE}")
    if DEVICE == "cpu":
        print("[train] WARNING: no GPU detected. Fine-tuning on CPU will be very slow -- "
              "confirm Colab's runtime is set to GPU (Runtime > Change runtime type).")

    train_dataset = ImageFolder(TRAIN_DIR, transform=preprocess)
    val_dataset = ImageFolder(VAL_DIR, transform=preprocess)

    # HARD STOP, not a warning: if the folder order doesn't match CLASSES,
    # the model's output index N will be silently mislabeled during
    # inference. This is the exact train/inference skew v4 Section 8 flags
    # as a risk -- it must be caught here, before training wastes GPU time.
    print(f"[train] ImageFolder classes (alphabetical): {train_dataset.classes}")
    print(f"[train] model_api.CLASSES (expected order): {CLASSES}")
    assert train_dataset.classes == CLASSES, (
        f"FATAL: class order mismatch. ImageFolder found {train_dataset.classes} "
        f"but model_api.CLASSES is {CLASSES}. Fix your patch_dataset folder names "
        f"(they must be exactly: {CLASSES}) before training -- do not proceed, "
        f"the trained model's outputs would be silently mislabeled."
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(NUM_CLASSES).to(DEVICE)  # NOT frozen -- v4 fine-tunes everything

    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print(f"[train] class weights: {dict(zip(train_dataset.classes, class_weights.tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(f"[train] epoch {epoch+1}/{MAX_EPOCHS} "
              f"loss={running_loss/len(train_loader):.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} "
              f"val_f1={val_metrics['weighted_f1']:.4f} "
              f"({elapsed:.1f}s)")

        if val_metrics["weighted_f1"] > best_f1:
            best_f1 = val_metrics["weighted_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT_PATH)
            print(f"[train]   -> new best (val_f1={best_f1:.4f}), saved to {OUTPUT_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[train] early stopping: no improvement for {PATIENCE} epochs")
                break

    print(f"\n[train] DONE. Best val weighted F1: {best_f1:.4f}")
    print(f"[train] Best weights saved to: {OUTPUT_PATH}")
    print(f"[train] NEXT STEP: upload this file to HDFS at hdfs:///models/rdamage_mobilenetv2.pt")
    print(f"[train]   e.g.: hdfs dfs -put -f {OUTPUT_PATH} /models/rdamage_mobilenetv2.pt")


if __name__ == "__main__":
    train()