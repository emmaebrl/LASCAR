# train_unet_pretrained.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import json


# ========== Config ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
IMG_SIZE = 256
EPOCHS = 5
NUM_CLASSES = 10  # Seulement classes valides (sans clouds et no_data)

CLASSES = [
    "no_data",
    "clouds",
    "artificial",
    "cultivated",
    "broadleaf",
    "coniferous",
    "herbaceous",
    "natural",
    "snow",
    "water",
]


# ========== Utils ==========
def load_tif(path):

    with rasterio.open(path) as src:
        arr = src.read()
        return np.transpose(arr, (1, 2, 0))  # (H, W, C)


# ========== Dataset ==========
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_tif(self.image_paths[idx])
        mask = load_tif(self.mask_paths[idx])[:, :, 0]  # mask 2D

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return img, mask.long()


# ========== Transforms ==========
train_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ]
)

# ========== Données ==========
CSV_PATH = "dataset/train_labels_GY1QjFw.csv"
IMG_DIR = "dataset/train/images"
MASK_DIR = "dataset/train/masks"

TEST_IMG_DIR = r"dataset\test\images"
TEST_CSV_PATH = r"dataset\test_images_kkwOpBC.csv"

df = pd.read_csv(CSV_PATH)

with open("splits/train_val_ids.json", "r") as f:
    split_data = json.load(f)
train_ids = split_data["train"]
val_ids = split_data["val"]
train_imgs = [os.path.join(IMG_DIR, f"{sid}.tif") for sid in train_ids]
val_imgs = [os.path.join(IMG_DIR, f"{sid}.tif") for sid in val_ids]
train_masks = [os.path.join(MASK_DIR, f"{sid}.tif") for sid in train_ids]
val_masks = [os.path.join(MASK_DIR, f"{sid}.tif") for sid in val_ids]


train_ds = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_ds = SegmentationDataset(val_imgs, val_masks, transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# ========== Modèle ==========
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,
    classes=NUM_CLASSES,
).to(DEVICE)

# ========== Entraînement ==========
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            out = model(img)
            loss = criterion(out, mask)
            val_loss += loss.item()
    return val_loss / len(val_loader)


print("🔧 Entraînement du modèle U-Net avec ResNet34...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for img, mask in loop:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    val_loss = validate(model, val_loader, criterion)
    print(
        f"✅ Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f} | Val Loss = {val_loss:.4f}"
    )


# ========== Sauvegarde ==========
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/unet_resnet34_finetuned.pth")
print("📦 Modèle sauvegardé sous: models/unet_resnet34_finetuned.pth")


# ================== POST-PROCESSING ==================
def mask_to_proportions(mask_pred, num_classes):
    flat = mask_pred.flatten()
    props = [(flat == i).sum() / len(flat) for i in range(num_classes)]
    return props


# ================== METRIC - KL Divergence on Val ==================
def kl_divergence(y_true, y_pred, eps=1e-8):
    y_true = np.clip(y_true, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    return np.sum(y_true * np.log(y_true / y_pred))


model.eval()
kl_scores = []
with torch.no_grad():
    for img, mask in tqdm(val_loader, desc="Evaluating KL"):
        img = img.to(DEVICE)
        pred = model(img)
        pred_mask = torch.argmax(pred.squeeze(0), dim=0).cpu().numpy()
        true_mask = mask.squeeze(0).numpy()

        pred_prop = mask_to_proportions(pred_mask, NUM_CLASSES)
        true_prop = mask_to_proportions(true_mask, NUM_CLASSES)

        kl = kl_divergence(np.array(true_prop), np.array(pred_prop))
        kl_scores.append(kl)

avg_kl_seg = np.mean(kl_scores)
print(f"\n🔍 KL Divergence on segmentation val set: {avg_kl_seg:.6f}")


### PREDICT ON TEST ###
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_tif(self.image_paths[idx])
        sample_id = os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        return img, sample_id


### MODIF DEBUT - Moyenne sur softmax (proportions plus lisses) ###
def predict_on_test(model, test_loader, num_classes):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for img, sample_id in tqdm(test_loader, desc="Predicting on test set"):
            img = img.to(DEVICE)
            out = model(img)
            soft = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # [C, H, W]
            proportions = soft.reshape(num_classes, -1).mean(axis=1)
            all_preds.append(proportions)
            all_ids.append(sample_id[0])

    return all_ids, all_preds


### MODIF FIN ###
# === Préparer test set ===
test_df = pd.read_csv(TEST_CSV_PATH)  # fichier contenant les sample_id
test_image_paths = [
    os.path.join(TEST_IMG_DIR, f"{str(s)}.tif") for s in test_df["sample_id"]
]
test_ds = TestDataset(test_image_paths, transform=train_transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# === Prédire ===
sample_ids, preds = predict_on_test(model, test_loader, num_classes=NUM_CLASSES)

# Re-normalisation des prédictions sans no_data/clouds si jamais inclus
preds = np.array(preds)
preds = preds / preds.sum(axis=1, keepdims=True)  # Juste au cas où

# Sauvegarde
df_sub = pd.DataFrame(preds, columns=CLASSES)
df_sub.insert(0, "sample_id", sample_ids)
df_sub.to_csv("submission_segmentation_unet.csv", index=False)
print("✅ Sauvegarde dans submission_segmentation.csv terminée.")
