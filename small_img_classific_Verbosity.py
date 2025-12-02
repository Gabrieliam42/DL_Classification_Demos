# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

import os, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageFile, features

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = r"D:\AI_Research\DL\imgdata"
CHECKPOINT_DIR = r"D:\AI_Research\DL\imgcheckpoints"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2
WORKERS = 4
PRINT_EVERY_BATCH = 1
USE_PRETRAINED = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def find_image_files(root: Path):
    files = []
    log(f"Searching for image files in {root}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
            log(f"  Found image: {p}")
    log(f"Total images found: {len(files)}")
    return sorted(files)

class FlatImageDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        log(f"FlatImageDataset created with {len(files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0

def prepare_datasets(data_dir):
    data_path = Path(data_dir)
    log(f"Preparing datasets from {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    webp_ok = features.check("webp")
    if not webp_ok:
        log("WARNING: Pillow compiled without WEBP support. .webp files may fail to open.")

    subdirs = [p for p in data_path.iterdir() if p.is_dir()]
    has_subfolders_with_images = False
    classes = []
    for d in subdirs:
        imgs = find_image_files(d)
        if imgs:
            has_subfolders_with_images = True
            classes.append(d.name)

    if has_subfolders_with_images:
        log("Detected class subfolders. Using ImageFolder dataset.")
        transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
        class_names = dataset.classes
        total_files = len(dataset)
        log(f"Classes detected: {class_names}, total images: {total_files}")
    else:
        files = [p for p in data_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not files:
            files = find_image_files(data_path)
        if not files:
            raise FileNotFoundError(f"No image files found under {data_dir}.")
        log(f"Flat folder detected with {len(files)} images")
        transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.1)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        dataset = FlatImageDataset(files, transform=transform)
        class_names = ["class0"]
        total_files = len(dataset)

    val_size = int(math.floor(VAL_SPLIT * total_files))
    train_size = total_files - val_size
    if val_size == 0:
        log("Validation split is 0, using train-only run")
        return dataset, None, class_names
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    log(f"Train size: {train_size}, Val size: {val_size}")
    return train_dataset, val_dataset, class_names

def build_model(num_classes, device):
    log(f"Building model with {num_classes} output classes")
    weights = ResNet18_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
    try:
        model = resnet18(weights=weights)
    except Exception as e:
        log(f"Failed to load pretrained weights: {e}, using untrained model")
        model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    log(f"Model on device: {device}")
    return model

def save_checkpoint(state, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(state, path)
    log(f"Checkpoint saved: {path}")

def main():
    log(f"Current working directory: {os.getcwd()}")
    log(f"Data folder: {DATA_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device info: {device}, torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")

    train_ds, val_ds, class_names = prepare_datasets(DATA_DIR)
    num_classes = len(class_names)

    log(f"Building dataloaders...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True) if val_ds else None
    log(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader) if val_loader else 0}")

    model = build_model(num_classes, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters: total={total_params}, trainable={trainable}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    log("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = running_correct = running_total = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (preds == yb).sum().item()
            running_total += xb.size(0)

            log(f"Epoch {epoch} batch {batch_idx}/{len(train_loader)}: "
                f"batch_loss={loss.item():.4f}, batch_acc={(preds==yb).float().mean().item():.4f}, "
                f"running_loss={running_loss/running_total:.4f}, running_acc={running_correct/running_total:.4f}")

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        epoch_time = time.time() - epoch_start
        log(f"Epoch {epoch} complete: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}, time={epoch_time:.1f}s")

        if val_loader:
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    preds = logits.argmax(dim=1)
                    val_loss += loss.item() * xb.size(0)
                    val_correct += (preds==yb).sum().item()
                    val_total += xb.size(0)
                    log(f"  Val batch: loss={loss.item():.4f}, acc={(preds==yb).float().mean().item():.4f}")

            val_loss /= val_total
            val_acc = val_correct / val_total
            log(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}, samples={val_total}")

        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "num_classes": num_classes,
            "class_names": class_names,
            "loss": epoch_loss,
            "acc": epoch_acc,
        }, epoch, CHECKPOINT_DIR)

    log("Training finished. Final evaluation:")
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.to(device)
        sample_logits = torch.softmax(model(sample_x), dim=1)
        for i in range(min(6, sample_x.size(0))):
            top1 = sample_logits[i].argmax().item()
            conf = sample_logits[i, top1].item()
            log(f"  Sample {i}: true={sample_y[i]}, pred_class={top1}, conf={conf:.3f}")

if __name__ == "__main__":
    main()
    
# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42