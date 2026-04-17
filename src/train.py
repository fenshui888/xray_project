import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import BaselineCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryImageFolder(Dataset):
    """Convert ImageFolder classes to NORMAL(0) vs PNEUMONIA(1)."""

    def __init__(self, root: str, transform=None):
        self.base = datasets.ImageFolder(root=root, transform=transform)
        self.class_names = self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        class_name = self.class_names[label]
        binary_label = 0 if class_name == "NORMAL" else 1
        return image, torch.tensor(binary_label, dtype=torch.float32)


def build_loaders(train_dir: str, val_dir: str, batch_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_ds = BinaryImageFolder(root=train_dir, transform=train_tfms)
    val_ds = BinaryImageFolder(root=val_dir, transform=eval_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, train_ds.class_names


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        losses.append(loss.item())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    pred_labels = [1 if p >= 0.5 else 0 for p in all_probs]
    f1 = f1_score(all_labels, pred_labels, pos_label=1)
    roc_auc = roc_auc_score(all_labels, all_probs)
    return float(np.mean(losses)), float(f1), float(roc_auc)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_loaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
    )

    model = BaselineCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_pneumonia": [],
        "val_roc_auc": [],
    }
    best_f1 = -1.0
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_f1, val_auc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_pneumonia"].append(val_f1)
        history["val_roc_auc"].append(val_auc)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f} val_auc={val_auc:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": 224,
                "mean": [0.5],
                "std": [0.5],
                "threshold": 0.5,
            }
            torch.save(checkpoint, output_path)
            print(f"Saved new best model to {output_path}")

    if args.history_out:
        history_path = Path(args.history_out)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Saved history to {history_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="models/baseline.pt")
    parser.add_argument("--history-out", default="outputs/baseline_history.json")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
