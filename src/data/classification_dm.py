#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/concrete-crack-cnn-vs-vit
#
#  This work is licensed under the MIT License.
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2026
#  Package:   concrete-crack-cnn-vs-vit — CNN vs ViT Benchmark

"""Lightning DataModule for SDNET2018 binary crack classification.

A DataModule is the PyTorch Lightning equivalent of a tf.data pipeline builder:
- __init__  = configure paths, batch size, image size (like setting up tf.data params)
- setup()   = load split file and create Dataset objects (like tf.data.Dataset.from_tensor_slices())
- *_dataloader() = wrap Datasets in DataLoaders (like .batch().prefetch())

DataLoaders handle batching, shuffling, and multi-process data loading.
The num_workers parameter spawns background processes to load data — this is
equivalent to tf.data.Dataset.prefetch(tf.data.AUTOTUNE) but explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.classification_dataset import CrackClassificationDataset

# Labels derived from directory name: C* = cracked (1), U* = uncracked (0)
_CRACKED_PREFIXES = {"CD", "CP", "CW"}

# ImageNet normalization — standard for pretrained timm models
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _label_from_path(rel_path: str) -> int:
    """Extract label from a relative path like 'D/CD/7001-115.jpg'.

    The second path component (CD, UD, CP, UP, CW, UW) encodes the class.
    """
    parent = Path(rel_path).parent.name
    return 1 if parent in _CRACKED_PREFIXES else 0


def _build_val_transforms(image_size: int) -> A.Compose:
    """Resize + Normalize only — no random augmentation for val/test."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _build_train_transforms(image_size: int) -> A.Compose:
    """Default training transforms: Resize + HorizontalFlip + Normalize.

    This is a minimal baseline. DATA-7 will replace this with configurable
    augmentation presets (light/medium/heavy).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class CrackClassificationDM(LightningDataModule):
    """Lightning DataModule for SDNET2018 crack classification.

    Loads a pre-computed split file (from DATA-4) and wraps each split in a
    CrackClassificationDataset with appropriate transforms.

    Keras/TF equivalent:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir, validation_split=0.15, subset="training", image_size=(224, 224))
        But here we get a proper train/val/test split with stratification,
        albumentations augmentation, and multi-worker loading.

    Args:
        data_dir: Path to assets/sdnet2018/ containing D/, P/, W/.
        split_file: Path to the JSON split file with train/val/test path lists.
        batch_size: Samples per batch (default 32).
        num_workers: Background worker processes for data loading (default 4).
            Like tf.data.prefetch(AUTOTUNE) but with explicit parallelism.
        image_size: Target image size in pixels (default 224, the timm standard).
    """

    def __init__(
        self,
        data_dir: str | Path = "assets/sdnet2018",
        split_file: str | Path = "configs/splits/sdnet2018_split.json",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Populated in setup()
        self._train_dataset: CrackClassificationDataset | None = None
        self._val_dataset: CrackClassificationDataset | None = None
        self._test_dataset: CrackClassificationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load split file, resolve paths, create Dataset objects.

        Called by Lightning's Trainer before training/validation/testing.
        The stage parameter tells us which datasets to prepare:
        - "fit" → train + val
        - "test" → test
        - None → all
        """
        with self.split_file.open() as f:
            splits: dict[str, list[str]] = json.load(f)

        if stage in ("fit", None):
            train_paths = [self.data_dir / p for p in splits["train"]]
            train_labels = [_label_from_path(p) for p in splits["train"]]
            self._train_dataset = CrackClassificationDataset(
                file_paths=train_paths,
                labels=train_labels,
                transform=_build_train_transforms(self.image_size),
            )

            val_paths = [self.data_dir / p for p in splits["val"]]
            val_labels = [_label_from_path(p) for p in splits["val"]]
            self._val_dataset = CrackClassificationDataset(
                file_paths=val_paths,
                labels=val_labels,
                transform=_build_val_transforms(self.image_size),
            )

        if stage in ("test", None):
            test_paths = [self.data_dir / p for p in splits["test"]]
            test_labels = [_label_from_path(p) for p in splits["test"]]
            self._test_dataset = CrackClassificationDataset(
                file_paths=test_paths,
                labels=test_labels,
                transform=_build_val_transforms(self.image_size),
            )

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader with shuffle=True and multi-worker loading.

        shuffle=True randomizes sample order each epoch — equivalent to
        tf.data.Dataset.shuffle(buffer_size).
        pin_memory=True speeds up CPU→GPU transfer.
        persistent_workers=True keeps worker processes alive between epochs.
        """
        assert self._train_dataset is not None, "Call setup('fit') first"
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup('fit') first"
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup('test') first"
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
