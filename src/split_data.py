import os
import shutil
import random
from pathlib import Path

def split_train_val(data_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Создает validation набор из train, сохраняя raw/train неизменным.
    Копирует файлы в output_dir/val с фиксированным seed для reproducibility.

    Параметры:
        data_dir(str): путь к data/raw
        output_dir(str): путь к data/interim
        val_ratio(float): доля изображений для валидации
        seed(int): фиксирует случайность
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    random.seed(seed)

    train_dir = Path(data_dir) / "train"
    val_dir = Path(output_dir) / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    os.makedirs(val_dir, exist_ok=True)

    for cls in os.listdir(train_dir):
        cls_train_path = train_dir / cls
        if not cls_train_path.is_dir():
            continue

        cls_val_path = val_dir / cls
        os.makedirs(cls_val_path, exist_ok=True)

        images = [
            img_name
            for img_name in os.listdir(cls_train_path)
            if (cls_train_path / img_name).is_file()
        ]
        random.shuffle(images)

        val_size = int(len(images) * val_ratio)
        val_images = images[:val_size]

        for img in val_images:
            src = cls_train_path / img
            dst = cls_val_path / img
            shutil.copy(src, dst)  # copy, no move!


if __name__ == "__main__":
    split_train_val("data/raw", "data/interim", val_ratio=0.2, seed=42)
    print("Validation split created in data/interim/val")
