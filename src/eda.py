import os
import argparse
from pathlib import Path
from PIL import Image
import random


def count_images(data_dir):
    """Подсчет количества изображений по split и классам"""
    stats = {}
    for split in os.listdir(data_dir):
        split_path = Path(data_dir) / split
        if not split_path.is_dir():
            continue
        stats[split] = {}
        for cls in os.listdir(split_path):
            cls_path = split_path / cls
            if not cls_path.is_dir():
                continue
            stats[split][cls] = len([p for p in cls_path.iterdir() if p.is_file()])
    return stats


def show_samples(data_dir, n=5, seed=42):
    """Показать случайные n изображений из каждого класса с фиксированным seed"""
    import matplotlib.pyplot as plt

    random.seed(seed)

    for split in os.listdir(data_dir):
        split_path = Path(data_dir) / split
        if not split_path.is_dir():
            continue

        for cls in os.listdir(split_path):
            cls_path = split_path / cls
            if not cls_path.is_dir():
                continue

            images = list(cls_path.iterdir())
            random.shuffle(images)
            images = images[:n]

            fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
            fig.suptitle(f"{split.upper()} - {cls}", fontsize=14)

            for i, img_path in enumerate(images):
                try:
                    img = Image.open(img_path)
                    axes[i].imshow(img, cmap="gray")
                    axes[i].axis("off")
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            plt.show()
            # показываем только один класс за раз
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/interim",
        help="Path to dataset root with split folders (default: data/interim)",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of sample images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-samples",
        action="store_true",
        help="Print only dataset statistics without plotting image samples",
    )
    args = parser.parse_args()

    data_path = args.data_path
    stats = count_images(data_path)
    for split, classes in stats.items():
        print(f"\n{split.upper()}")
        for cls, count in classes.items():
            print(f"{cls}: {count}")

    if not args.skip_samples:
        # показать несколько случайных образцов
        show_samples(data_path, n=args.n, seed=args.seed)
