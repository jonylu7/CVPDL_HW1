import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
# from tqdm import tqdm

# ------------------------------
# 1. 資料集路徑
# ------------------------------
# Update paths to access files from Google Drive
TRAIN_IMG_DIR = '/content/drive/My Drive/CVPDL_datasets/train/img'
TRAIN_GT_FILE = '/content/drive/My Drive/CVPDL_datasets/train/gt.txt'
TEST_IMG_DIR  = '/content/drive/My Drive/CVPDL_datasets/test/img'

# ------------------------------
# 2. Dataset
# ------------------------------
class PigDataset(Dataset):
    def __init__(self, img_dir, gt_file=None, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = sorted(os.listdir(img_dir))

        self.boxes = {}
        if gt_file:
            with open(gt_file) as f:
                for line in f:
                    line = line.strip().split(',')
                    if len(line) < 5:
                        continue
                    img_id = line[0]
                    x, y, w, h = map(float, line[1:5])
                    if w <= 0 or h <= 0:
                        continue
                    if img_id not in self.boxes:
                        self.boxes[img_id] = []
                    self.boxes[img_id].append([x, y, x + w, y + h])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img)

        img_id = str(int(os.path.splitext(img_name)[0]))
        boxes = self.boxes.get(img_id, [])
        if len(boxes) == 0:
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target, img_name

# ------------------------------
# 3. DataLoader + Validation Split
# ------------------------------
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))

full_dataset = PigDataset(TRAIN_IMG_DIR, TRAIN_GT_FILE)
n_total = len(full_dataset)
n_val = int(0.2 * n_total)
n_train = n_total - n_val

train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

# ------------------------------
# 4. Faster R-CNN + pretrained weights
# ------------------------------
device = torch.device('cuda')

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
model = fasterrcnn_resnet50_fpn_v2(weights="COCO_V1")

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# ------------------------------
# 5. 訓練設定
# ------------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)
num_epochs = 10

# ------------------------------
# 6. 訓練 + 驗證 Loop
# ------------------------------
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    if epoch == 9:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        print("Learning Rate scaled down")

    # for imgs, targets, _ in tqdm(train_loader, desc="Training"):
    for imgs, targets, _ in train_loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        running_loss += losses.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f}")

    # ------------------------------
    # Validation
    # ------------------------------
    val_loss = 0.0
    model.train()  # enable loss output (but no backward)
    with torch.no_grad():
        # for imgs, targets, _ in tqdm(val_loader, desc="Validating"):
        for imgs, targets, _ in val_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    model.eval()  # back to eval mode for next epoch

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

# ------------------------------
# 7. 預測 & submission
# ------------------------------
model.eval()
test_imgs = sorted(os.listdir(TEST_IMG_DIR))
predictions = []

with torch.no_grad():
    # for img_name in tqdm(test_imgs, desc="Testing"):
    for img_name in test_imgs:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        pred = model([img_tensor])[0]

        img_id = int(os.path.splitext(img_name)[0])
        parts = []
        for score, box in zip(pred['scores'], pred['boxes']):
            if score < 0.3:
                continue
            x_min, y_min, x_max, y_max = box.tolist()
            w, h = x_max - x_min, y_max - y_min
            parts.append(f"{score:.6f} {x_min:.2f} {y_min:.2f} {w:.2f} {h:.2f} 0")

        pred_str = " ".join(parts)
        predictions.append([img_id, pred_str])

submission = pd.DataFrame(predictions, columns=['Image_ID', 'PredictionString'])
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission saved: submission.csv")