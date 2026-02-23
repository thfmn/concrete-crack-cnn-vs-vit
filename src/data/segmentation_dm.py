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

"""Lightning DataModule for CrackSeg9k semantic segmentation.

Same interface as CrackClassificationDM — a DataModule is the PyTorch Lightning
equivalent of a tf.data pipeline builder:
- __init__  = configure paths, batch size, image size
- setup()   = discover image-mask pairs, create/load split, build Datasets
- *_dataloader() = wrap Datasets in DataLoaders (like .batch().prefetch())

CrackSeg9k has a flat structure:
    assets/crackseg9k/images/*.png   (9,159 RGB images)
    assets/crackseg9k/masks/Masks/*.png  (9,159 grayscale masks)

The split is 80/10/10 (train/val/test), saved as JSON for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.augmentation import Preset, get_train_transforms, get_val_transforms
from src.data.segmentation_dataset import CrackSegmentationDataset

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _discover_pairs(data_dir: Path) -> list[str]:
    """Find all image filenames that have a corresponding mask.

    Returns sorted list of filenames (e.g. ["a_0_10.png", "a_0_11.png", ...])
    that exist in both images/ and masks/Masks/.
    """
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks" / "Masks"

    image_names = {f.name for f in images_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS}
    mask_names = {f.name for f in masks_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS}

    # Only keep filenames present in both directories
    paired = sorted(image_names & mask_names)
    if not paired:
        msg = f"No image-mask pairs found in {data_dir}"
        raise ValueError(msg)

    n_orphan_images = len(image_names - mask_names)
    n_orphan_masks = len(mask_names - image_names)
    if n_orphan_images or n_orphan_masks:
        logger.warning(
            "Orphans: %d images without masks, %d masks without images",
            n_orphan_images,
            n_orphan_masks,
        )

    return paired


def create_segmentation_splits(
    data_dir: str | Path,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create random 80/10/10 train/val/test splits for CrackSeg9k.

    Unlike classification, there's no class label to stratify on — we just
    do a random shuffle split. The split stores filenames (not full paths)
    so it's portable across machines.

    Args:
        data_dir: Path to assets/crackseg9k/.
        val_ratio: Fraction for validation (default 0.10).
        test_ratio: Fraction for test (default 0.10).
        seed: Random seed for reproducibility.

    Returns:
        Dict with "train", "val", "test" keys mapping to lists of filenames.
    """
    data_dir = Path(data_dir)
    filenames = _discover_pairs(data_dir)

    # Two-step split: separate test first, then split remainder into train/val
    val_test_ratio = val_ratio + test_ratio
    train_names, valtest_names = train_test_split(
        filenames,
        test_size=val_test_ratio,
        random_state=seed,
    )

    test_fraction = test_ratio / val_test_ratio
    val_names, test_names = train_test_split(
        valtest_names,
        test_size=test_fraction,
        random_state=seed,
    )

    splits = {
        "train": sorted(train_names),
        "val": sorted(val_names),
        "test": sorted(test_names),
    }

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    return splits


class CrackSegmentationDM(LightningDataModule):
    """Lightning DataModule for CrackSeg9k crack segmentation.

    Discovers image-mask pairs in the CrackSeg9k directory, creates (or loads)
    an 80/10/10 split, and wraps each split in a CrackSegmentationDataset with
    appropriate transforms.

    Keras/TF equivalent:
        In Keras you'd use tf.data.Dataset.from_tensor_slices((images, masks))
        with .map(augment).batch().prefetch(). Here, the Dataset handles
        per-sample loading + augmentation, and the DataLoader handles batching
        + multi-process prefetching.

    Args:
        data_dir: Path to assets/crackseg9k/ containing images/ and masks/Masks/.
        split_file: Path to save/load the JSON split file. Generated automatically
            on first run if it doesn't exist.
        batch_size: Samples per batch (default 8 — segmentation uses more memory).
        num_workers: Background worker processes for data loading (default 4).
        image_size: Target image size in pixels (default 224).
        aug_preset: Augmentation strength — "light", "medium", or "heavy".
        val_ratio: Fraction for validation split (default 0.10).
        test_ratio: Fraction for test split (default 0.10).
        seed: Random seed for split reproducibility (default 42).
    """

    def __init__(
        self,
        data_dir: str | Path = "assets/crackseg9k",
        split_file: str | Path = "configs/splits/crackseg9k_split.json",
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 224,
        aug_preset: Preset = "light",
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split_file = Path(split_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.aug_preset = aug_preset
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # Populated in setup()
        self._train_dataset: CrackSegmentationDataset | None = None
        self._val_dataset: CrackSegmentationDataset | None = None
        self._test_dataset: CrackSegmentationDataset | None = None

    def _resolve_paths(self, filenames: list[str]) -> tuple[list[Path], list[Path]]:
        """Convert filenames to absolute image and mask paths."""
        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks" / "Masks"
        image_paths = [images_dir / f for f in filenames]
        mask_paths = [masks_dir / f for f in filenames]
        return image_paths, mask_paths

    def setup(self, stage: str | None = None) -> None:
        """Load or create split, then build Dataset objects.

        If the split file exists, loads it. Otherwise, discovers pairs,
        creates an 80/10/10 split, and saves it to disk for reproducibility.

        Called by Lightning's Trainer before training/validation/testing.
        """
        # Load existing split or create a new one
        if self.split_file.exists():
            with self.split_file.open() as f:
                splits: dict[str, list[str]] = json.load(f)
        else:
            splits = create_segmentation_splits(
                self.data_dir,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                seed=self.seed,
            )
            self.split_file.parent.mkdir(parents=True, exist_ok=True)
            with self.split_file.open("w") as f:
                json.dump(splits, f, indent=2)
            logger.info("Saved segmentation split to %s", self.split_file)

        if stage in ("fit", None):
            train_imgs, train_masks = self._resolve_paths(splits["train"])
            self._train_dataset = CrackSegmentationDataset(
                image_paths=train_imgs,
                mask_paths=train_masks,
                transform=get_train_transforms(self.aug_preset, self.image_size),
            )

            val_imgs, val_masks = self._resolve_paths(splits["val"])
            self._val_dataset = CrackSegmentationDataset(
                image_paths=val_imgs,
                mask_paths=val_masks,
                transform=get_val_transforms(self.image_size),
            )

        if stage in ("test", None):
            test_imgs, test_masks = self._resolve_paths(splits["test"])
            self._test_dataset = CrackSegmentationDataset(
                image_paths=test_imgs,
                mask_paths=test_masks,
                transform=get_val_transforms(self.image_size),
            )

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader with shuffle and multi-worker loading.

        shuffle=True randomizes sample order each epoch (like tf.data.shuffle).
        pin_memory=True speeds up CPU→GPU transfer.
        persistent_workers=True keeps workers alive between epochs.
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
