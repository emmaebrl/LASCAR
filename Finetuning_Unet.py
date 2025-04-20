import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# ========== Configuration ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
BATCH_SIZE = 8
EPOCHS = 20

TRAIN_IMG_DIR = r"data/train/images"
TRAIN_MASK_DIR = r"data/train/masks"
PROPORTION_CSV = r"data/train_labels_GY1QjFw.csv"
TEST_IMG_DIR = r"data/test/images"
TEST_CSV_PATH = r"data/test_images_kkwOpBC.csv"


# ========== Utils ==========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def load_tif(path):
    path = os.path.normpath(path)
    with rasterio.open(path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
    return img


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
        mask = load_tif(self.mask_paths[idx])[:, :, 0]  # 2D mask
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        return img, mask.long()


# ========== Transforms ==========
train_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406, 0.5), std=(0.229, 0.224, 0.225, 0.25)),
        ToTensorV2(),
    ]
)

# ========== Load Data ==========
df = pd.read_csv(PROPORTION_CSV)
image_paths = [os.path.join(TRAIN_IMG_DIR, f"{str(f)}.tif") for f in df["sample_id"]]
mask_paths = [os.path.join(TRAIN_MASK_DIR, f"{str(f)}.tif") for f in df["sample_id"]]

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)
train_ds = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_ds = SegmentationDataset(val_imgs, val_masks, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# ========== Model ==========
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=4,
    classes=NUM_CLASSES,
)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5
)
dice_loss = DiceLoss(mode="multiclass", ignore_index=0)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)


def combined_loss(pred, target):
    return 0.7 * ce_loss(pred, target) + 0.3 * dice_loss(pred, target)


# ========== Training ==========
print("Training U-Net model...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for img, mask in loop:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = combined_loss(out, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# ========== Save model ==========
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/unet_resnet50_best.pth")


# ========== Post-processing ==========
def mask_to_proportions(mask_pred, num_classes):
    flat = mask_pred.flatten()
    props = [(flat == i).sum() / len(flat) for i in range(num_classes)]
    return props


def kl_divergence(y_true, y_pred, eps=1e-8):
    y_true = np.clip(y_true, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    return np.sum(y_true * np.log(y_true / y_pred))


# ========== Evaluation ==========
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


# ========== Predict on Test ==========
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


def predict_on_test(model, test_loader, num_classes):
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for img, sample_id in tqdm(test_loader, desc="Predicting on test set"):
            img = img.to(DEVICE)
            out = model(img)
            soft = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
            proportions = soft.reshape(num_classes, -1).mean(axis=1)
            all_preds.append(proportions)
            all_ids.append(sample_id[0])
    return all_ids, all_preds


# ========== Test Submission ==========
test_df = pd.read_csv(TEST_CSV_PATH)
test_image_paths = [
    os.path.join(TEST_IMG_DIR, f"{str(s)}.tif") for s in test_df["sample_id"]
]
test_ds = TestDataset(test_image_paths, transform=train_transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

model.load_state_dict(torch.load("models/unet_resnet50_best.pth"))
model.eval()
sample_ids, preds = predict_on_test(model, test_loader, num_classes=NUM_CLASSES)

columns = [
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
preds = np.array(preds)
preds = preds / preds.sum(axis=1, keepdims=True)
df_sub = pd.DataFrame(preds, columns=columns)
df_sub.insert(0, "sample_id", sample_ids)
df_sub.to_csv("submission.csv", index=False)
print("✅ Sauvegarde dans submission.csv terminée.")
