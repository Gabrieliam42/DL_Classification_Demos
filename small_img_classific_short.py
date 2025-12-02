# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

import os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = r"D:\AI_Research\DL\imgdata"
CHECKPOINT_DIR = r"D:\AI_Research\DL\imgcheckpoints"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2
WORKERS = 4
USE_PRETRAINED = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = [p for p in os.listdir(folder) if os.path.splitext(p)[1].lower() in IMG_EXTS]
        self.files = [os.path.join(folder, f) for f in self.files]
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(1.0, dtype=torch.float32)

def prepare_data(folder):
    transform = transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE*1.1)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = FlatImageDataset(folder, transform=transform)
    n_val = int(len(dataset)*VAL_SPLIT)
    n_train = len(dataset)-n_val
    if n_val == 0:
        return dataset, None
    return random_split(dataset, [n_train, n_val])

def build_model(device):
    weights = ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)

def save_checkpoint(model, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt")
    torch.save(model.state_dict(), path)
    log(f"Saved checkpoint: {path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    train_ds, val_ds = prepare_data(DATA_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS) if val_ds else None

    model = build_model(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        log(f"Epoch {epoch}  loss: {running_loss/len(train_loader.dataset):.4f}")
        save_checkpoint(model, epoch)

if __name__ == "__main__":
    main()

# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42