from ast import Continue
import os
from collections import Counter


def count_images(data_dir):
    stats = {}

    for split in os.listdir(data_dir):
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        stats[split] = {}

        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            stats[split][cls] = len(os.listdir(cls_path))

    return stats


if __name__ == "__main__":
    data_path = "data/raw"
    stats = count_images(data_path)

    for split, classes in stats.items():
        print(f"\n{split.upper()}")
        for cls, count in classes.items():
            print(f"{cls}: {count}")
