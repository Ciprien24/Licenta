# train_vit_baseline.py
# Pure ViT baseline on BUSI using the EXACT SAME split as your CNN run (split.json).
# Saves the same artifacts: checkpoints/, preds/, plots/, config.json, split.json copy, reports.

import os
import json
import time
import random
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # only used if you choose to NOT reuse split

# ----------------------------
# CONFIG
# ----------------------------
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print("MPS available", torch.backends.mps.is_available())
BATCH_SIZE = 16

# Two-phase training (recommended for small datasets like BUSI)
EPOCHS_PHASE1 = 10   # head-only
EPOCHS_PHASE2 = 20   # fine-tune last blocks

LR_PHASE1 = 1e-3
LR_PHASE2 = 2e-5
WEIGHT_DECAY = 0.05

DATA_PATH = "./Dataset_BUSI_with_GT/"  # your BUSI root folder

# IMPORTANT: set this to the CNN run you want to reuse split from
# Example: "runs/cnn_busi_baseline_20260226_120305/split.json"
SPLIT_JSON_PATH = "runs/cnn_busi_baseline_20260225_162715/split.json"

IMG_SIZE = (224, 224)
SEED = 42

print(f"Using device: {DEVICE}")

# ----------------------------
# RUN FOLDER
# ----------------------------
RUN_NAME = time.strftime("vit_busi_baseline_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(os.path.join(RUN_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "preds"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "plots"), exist_ok=True)

# ----------------------------
# SEEDING
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ----------------------------
# DATASET
# ----------------------------
class BUSIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label, img_path

def load_data(root_dir):
    image_paths = []
    labels = []
    classes = ["normal", "benign", "malignant"]

    for label_idx, class_name in enumerate(classes):
        full_class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(full_class_path):
            raise FileNotFoundError(f"Class folder not found: {full_class_path}")

        for filename in os.listdir(full_class_path):
            if "mask" not in filename and filename.endswith(".png"):
                image_paths.append(os.path.join(full_class_path, filename))
                labels.append(label_idx)

    return image_paths, labels, classes

def load_split_or_make_one(split_json_path, data_path, seed=42):
    """
    If split_json_path exists, load it and use those exact splits.
    Otherwise, create a new 70/10/20 split (train/val/test) and return it.
    """
    if split_json_path and os.path.exists(split_json_path):
        with open(split_json_path, "r") as f:
            split = json.load(f)

        X_train = split["train"]
        X_val   = split["val"]
        X_test  = split["test"]
        y_train = split["y_train"]
        y_val   = split["y_val"]
        y_test  = split["y_test"]
        class_names = split.get("classes", ["normal", "benign", "malignant"])

        return X_train, X_val, X_test, y_train, y_val, y_test, class_names, True

    # fallback: make a new split (you usually don't want this for fair comparisons)
    all_images, all_labels, class_names = load_data(data_path)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        all_images, all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.125,  # 10% of total
        stratify=y_trainval,
        random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names, False

# ----------------------------
# LOAD SPLIT
# ----------------------------
print("Loading split...")
X_train, X_val, X_test, y_train, y_val, y_test, class_names, reused_split = load_split_or_make_one(
    SPLIT_JSON_PATH, DATA_PATH, seed=SEED
)

print(f"Reused split from CNN run: {reused_split}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Classes: {class_names}")

# Save/copy split into THIS run folder for reproducibility
split_to_save = {
    "train": X_train,
    "val": X_val,
    "test": X_test,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,
    "classes": class_names,
    "source_split_json": SPLIT_JSON_PATH if reused_split else None
}
with open(os.path.join(RUN_DIR, "split.json"), "w") as f:
    json.dump(split_to_save, f, indent=2)

# ----------------------------
# TRANSFORMS
# ----------------------------
# ViT expects ImageNet-like normalization when using ImageNet pretrained weights
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ----------------------------
# LOADERS
# ----------------------------
train_dataset = BUSIDataset(X_train, y_train, transform=train_transforms)
val_dataset   = BUSIDataset(X_val,   y_val,   transform=val_transforms)
test_dataset  = BUSIDataset(X_test,  y_test,  transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# MODEL (ViT)
# ----------------------------
print("Setting up ViT-B/16 (ImageNet pretrained)...")
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)

# Replace classification head to match 3 classes
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, len(class_names))
model = model.to(DEVICE)

# ----------------------------
# SAVE CONFIG
# ----------------------------
config = {
    "device": str(DEVICE),
    "batch_size": BATCH_SIZE,
    "epochs_phase1_head_only": EPOCHS_PHASE1,
    "epochs_phase2_finetune": EPOCHS_PHASE2,
    "lr_phase1": LR_PHASE1,
    "lr_phase2": LR_PHASE2,
    "weight_decay": WEIGHT_DECAY,
    "data_path": DATA_PATH,
    "seed": SEED,
    "img_size": list(IMG_SIZE),
    "model": "vit_b_16_imagenet",
    "pretrained_weights": "ViT_B_16_Weights.IMAGENET1K_V1",
    "split_reused_from": SPLIT_JSON_PATH if reused_split else None,
    "classes": class_names,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "train_aug": "RandomResizedCrop(0.85-1.0), HFlip, Rotation(10)",
}
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# ----------------------------
# TRAIN / EVAL HELPERS
# ----------------------------
criterion = nn.CrossEntropyLoss()

best_val_acc = -1.0
best_path = os.path.join(RUN_DIR, "checkpoints", "best.pt")
last_path = os.path.join(RUN_DIR, "checkpoints", "last.pt")

def train_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total

def validate_epoch(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total

def evaluate_and_save(model, loader, split_name="val"):
    model.eval()
    all_preds, all_labels, all_paths = [], [], []
    all_probs = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    with open(os.path.join(RUN_DIR, f"{split_name}_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    probs_df = pd.DataFrame(all_probs, columns=[f"prob_{c}" for c in class_names])
    df = pd.DataFrame({
        "path": all_paths,
        "y_true": all_labels,
        "y_pred": all_preds,
    })
    df = pd.concat([df, probs_df], axis=1)
    df.to_csv(os.path.join(RUN_DIR, "preds", f"{split_name}_predictions.csv"), index=False)

    return acc, cm, report

# ----------------------------
# PHASE 1: HEAD-ONLY TRAINING
# ----------------------------
print("\nPHASE 1: Train head only")

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.parameters():
    p.requires_grad = True

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(EPOCHS_PHASE1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    val_acc = validate_epoch(model, val_loader)

    # save last
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch + 1,
        "val_acc": val_acc,
        "phase": 1
    }, last_path)

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch + 1,
            "val_acc": val_acc,
            "phase": 1
        }, best_path)

    print(f"[P1] Epoch [{epoch+1}/{EPOCHS_PHASE1}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ----------------------------
# PHASE 2: FINETUNE LAST 2 ENCODER BLOCKS + HEAD
# ----------------------------
print("\nPHASE 2: Fine-tune last 2 encoder blocks + head")

# Unfreeze last 2 transformer blocks
for blk in model.encoder.layers[-2:]:
    for p in blk.parameters():
        p.requires_grad = True

# Keep head trainable too
for p in model.heads.parameters():
    p.requires_grad = True

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE2,
    weight_decay=WEIGHT_DECAY
)

for epoch in range(EPOCHS_PHASE2):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    val_acc = validate_epoch(model, val_loader)

    # save last
    torch.save({
        "model_state": model.state_dict(),
        "epoch": (EPOCHS_PHASE1 + epoch + 1),
        "val_acc": val_acc,
        "phase": 2
    }, last_path)

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "epoch": (EPOCHS_PHASE1 + epoch + 1),
            "val_acc": val_acc,
            "phase": 2
        }, best_path)

    print(f"[P2] Epoch [{epoch+1}/{EPOCHS_PHASE2}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")

# ----------------------------
# LOAD BEST + EVALUATE
# ----------------------------
ckpt = torch.load(best_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

# Evaluate on val and test (and save preds/reports)
evaluate_and_save(model, val_loader, "val")
evaluate_and_save(model, test_loader, "test")

print("Saved test report to:", os.path.join(RUN_DIR, "test_classification_report.txt"))
print("Saved test preds to :", os.path.join(RUN_DIR, "preds", "test_predictions.csv"))
print("Run directory:", RUN_DIR)