import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
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
def main(args):
    checkpoint = torch.load(args.model, map_location="cpu")
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
    dataset = BinaryImageFolder(root=args.input_dir, transform=tfms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BaselineCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    timings = []
    measured = 0

    for images, _ in loader:
        if measured >= args.num_batches:
            break
        start = time.perf_counter()
        _ = model(images)
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0 / images.shape[0]
        timings.append(elapsed_ms)
        measured += 1

    report = {
        "batch_size": args.batch_size,
        "num_batches_measured": measured,
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_p95": float(np.percentile(timings, 95)),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--report", default="outputs/baseline_latency.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
