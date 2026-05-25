"""
Eyes ROI Dataset
Loads closed / open crops from dataset/eyes/.
Label 0 = closed, Label 1 = open.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

_EYES_INPUT_H = 64
_EYES_INPUT_W = 64

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

_TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

_VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)


_MINORITY_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((_EYES_INPUT_H, _EYES_INPUT_W)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class EyesDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        class_to_label: dict[str, int],
        transform=None,
        minority_transform=None,
        minority_label: int = 0,
    ) -> None:
        self.transform = transform
        self.minority_transform = minority_transform
        self.minority_label = minority_label
        self.samples: list[tuple[Path, int]] = []

        root = Path(root)
        for class_name, label in class_to_label.items():
            class_dir = root / class_name
            img_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                img_paths.extend(class_dir.glob(ext))
            for img_path in sorted(img_paths):
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.minority_transform and label == self.minority_label:
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    root: str | Path,
    class_to_label: dict[str, int],
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    sampler = None,
    downsample_majority: bool = False,
) -> tuple[DataLoader, DataLoader]:
    import random
    root = Path(root)
    all_samples = EyesDataset(root, class_to_label, transform=None).samples

    if downsample_majority:
        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, (_, label) in enumerate(all_samples):
            label_to_indices[label].append(i)

        min_count = min(len(idxs) for idxs in label_to_indices.values())
        print(f"  Downsampling majority to {min_count} samples per class")

        balanced_indices = []
        for label, idxs in label_to_indices.items():
            random.seed(42)
            sampled = random.sample(idxs, min_count)
            balanced_indices.extend(sampled)

        all_samples = [all_samples[i] for i in balanced_indices]
        
    # --- Stratified split ---
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, (_, label) in enumerate(all_samples):
        label_to_indices[label].append(i)

    train_indices: list[int] = []
    val_indices:   list[int] = []

    for label, idxs in label_to_indices.items():
        idxs_tensor = torch.tensor(idxs)[torch.randperm(len(idxs))].tolist()
        n_val = max(1, int(len(idxs_tensor) * val_split))
        val_indices.extend(idxs_tensor[:n_val])
        train_indices.extend(idxs_tensor[n_val:])

    train_dataset = EyesDataset(
        root, class_to_label,
        transform=_TRAIN_TRANSFORMS,
        minority_transform=_MINORITY_TRANSFORMS,
    )
    val_dataset = EyesDataset(
        root, class_to_label,
        transform=_VAL_TRANSFORMS,
    )

    train_dataset.samples = [all_samples[i] for i in train_indices]
    val_dataset.samples   = [all_samples[i] for i in val_indices]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, train_indices