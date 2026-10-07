import os
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from src.config import (
    SPLITS_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    CLASS_TO_IDX,
    NUM_WORKERS
)
from src.enhancement import LeafEnhancementEngine

class AppleLeafDataset(Dataset):
    """
    PyTorch Dataset for AppleLeaf9 supporting raw and enhanced modes.
    """
    def __init__(
        self,
        csv_file: Path,
        mode: str = "enhanced",  # "raw" or "enhanced"
        transform: Optional[Callable] = None
    ):
        assert mode in ["raw", "enhanced"], f"mode must be 'raw' or 'enhanced', got '{mode}'"
        self.csv_file = Path(csv_file)
        self.mode = mode
        self.transform = transform
        self.df = pd.read_csv(self.csv_file)
        self.enhancement_engine = LeafEnhancementEngine() if mode == "enhanced" else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path = row["file_path"]
        label = int(row["label"])

        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Apply domain-specific enhancement if configured
        if self.mode == "enhanced" and self.enhancement_engine is not None:
            image = self.enhancement_engine(image)

        # Apply transforms (augmentation, resizing, normalization)
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Fallback default transform
            fallback_transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            image_tensor = fallback_transform(image)

        return image_tensor, label, img_path

def get_transforms(split: str = "train") -> T.Compose:
    """
    Generates standard torchvision transforms with agronomic augmentations.
    """
    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=25),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def compute_class_weights(train_csv_path: Optional[Path] = None) -> torch.Tensor:
    """
    Calculates balanced class weights: w_c = N / (K * N_c)
    Crucial for mitigating AppleLeaf9's 15.9x class imbalance.
    """
    if train_csv_path is None:
        train_csv_path = SPLITS_DIR / "train.csv"

    df = pd.read_csv(train_csv_path)
    class_counts = df["label"].value_counts().sort_index()
    total_samples = len(df)
    n_classes = NUM_CLASSES

    weights = []
    for c in range(n_classes):
        count = class_counts.get(c, 1)
        w = total_samples / (n_classes * count)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(
    mode: str = "enhanced",
    batch_size: int = 32,
    num_workers: int = NUM_WORKERS,
    splits_dir: Optional[Path] = None
) -> Dict[str, DataLoader]:
    """
    Creates DataLoaders for train, val, and test sets.
    """
    if splits_dir is None:
        splits_dir = SPLITS_DIR

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    train_dataset = AppleLeafDataset(
        csv_file=train_csv,
        mode=mode,
        transform=get_transforms("train")
    )

    val_dataset = AppleLeafDataset(
        csv_file=val_csv,
        mode=mode,
        transform=get_transforms("val")
    )

    test_dataset = AppleLeafDataset(
        csv_file=test_csv,
        mode=mode,
        transform=get_transforms("test")
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
    }

    return dataloaders
