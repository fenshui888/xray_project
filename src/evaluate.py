import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import BaselineCNN


class BinaryImageFolder(Dataset):
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


@torch.no_grad()
def run_eval(model, loader, device, threshold):
    model.eval()
    probs = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device)
        logits = model(images)
        batch_probs = torch.sigmoid(logits).cpu().numpy().tolist()
        probs.extend(batch_probs)
        labels.extend(batch_labels.numpy().tolist())

    preds = [1 if p >= threshold else 0 for p in probs]
    f1 = f1_score(labels, preds, pos_label=1)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return f1, auc, cm


def save_confusion_matrix(cm, out_path):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    checkpoint = torch.load(args.model, map_location="cpu")
    threshold = checkpoint.get("threshold", 0.5)
    image_size = checkpoint.get("image_size", 224)
    mean = checkpoint.get("mean", [0.5])
    std = checkpoint.get("std", [0.5])

    tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = BinaryImageFolder(root=args.val_dir, transform=tfms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    f1, auc, cm = run_eval(model, loader, device, threshold)
    save_confusion_matrix(cm, args.cm)

    report = {
        "f1_pneumonia": f1,
        "roc_auc": auc,
        "threshold": threshold,
        "num_samples": len(dataset),
        "confusion_matrix": cm,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved confusion matrix to {args.cm}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report", default="outputs/baseline_eval.json")
    parser.add_argument("--cm", default="outputs/baseline_cm.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
