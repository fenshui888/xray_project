import os
import shutil
import random


def split_train_val(data_dir, val_ratio=0.2):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    os.makedirs(val_dir, exist_ok=True)

    for cls in os.listdir(train_dir):
        cls_train_path = os.path.join(train_dir, cls)
        cls_val_path = os.path.join(val_dir, cls)

        os.makedirs(cls_val_path, exist_ok=True)

        images = os.listdir(cls_train_path)
        random.shuffle(images)

        val_size = int(len(images) * val_ratio)
        val_images = images[:val_size]

        for img in val_images:
            src = os.path.join(cls_train_path, img)
            dst = os.path.join(cls_val_path, img)

            shutil.move(src, dst)


if __name__ == "__main__":
    split_train_val("data/raw", val_ratio=0.2)
