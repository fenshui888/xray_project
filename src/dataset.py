from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transforms(img_size: int = 224):

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_transforms


def get_datasets(train_dir: str, val_dir: str, img_size: int = 224):

    train_tfms, val_tfms = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_tfms
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_tfms
    )

    return train_dataset, val_dataset


def get_dataloaders(
        batch_size: int = 32,
        img_size: int = 224,
        num_workers: int = 2
):

    train_dataset, val_dataset = get_datasets(
        train_dir="data/raw/train",      # ← use raw
        val_dir="data/interim/val",
        img_size=img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


if __name__ == "__main__":

    from pathlib import Path

    assert Path("data/raw/train").exists(), "Train path not found!"
    assert Path("data/interim/val").exists(), "Val path not found!"
    train_loader, val_loader = get_dataloaders()

    images, labels = next(iter(train_loader))
