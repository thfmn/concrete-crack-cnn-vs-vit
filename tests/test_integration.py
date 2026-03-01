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

"""Integration tests using real images from the dataset.

Unlike the unit tests (which use synthetic random pixels), these tests run
the full data pipeline on a small fixed set of real crack images to catch
issues that only appear with real-world data:

- JPEG decoding artefacts, unusual colour profiles
- Anti-aliased mask binarization on real crack edges
- Augmentation edge cases (very bright/dark regions, high contrast cracks)

Fixtures live in ``tests/fixtures/`` and are committed to git (~776 KB total):

- ``fixtures/sdnet2018/``: 12 images (2 per class dir), SDNET2018 layout
- ``fixtures/crackseg9k/``: 6 image+mask pairs, CrackSeg9k layout

Run integration tests:
    uv run pytest tests/test_integration.py -x -q

Skip in fast CI:
    uv run pytest tests/ -x -q -m "not integration"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.classification_dataset import CrackClassificationDataset
from src.data.classification_dm import CrackClassificationDM
from src.data.segmentation_dataset import CrackSegmentationDataset
from src.data.segmentation_dm import CrackSegmentationDM, create_segmentation_splits
from src.data.split import create_classification_splits

# All tests in this module require the real fixture images.
pytestmark = pytest.mark.integration

# Number of real images in the fixtures (2 per class × 6 classes).
_SDNET2018_N = 12
# Number of real image+mask pairs.
_CRACKSEG9K_N = 6


# ---------------------------------------------------------------------------
# Classification pipeline — real SDNET2018 images
# ---------------------------------------------------------------------------


class TestClassificationPipelineReal:
    """End-to-end classification pipeline on real SDNET2018 images."""

    def test_split_creation(self, real_sdnet2018_dir: Path, tmp_path: Path) -> None:
        """create_classification_splits discovers all 12 fixture images."""
        splits = create_classification_splits(
            real_sdnet2018_dir, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        total = sum(len(v) for v in splits.values())
        assert total == _SDNET2018_N

    def test_dataset_loads_real_images(self, real_sdnet2018_dir: Path) -> None:
        """CrackClassificationDataset returns valid tensors from real JPEGs."""
        paths = sorted(real_sdnet2018_dir.rglob("*.jpg"))
        labels = [1 if p.parent.name.startswith("C") else 0 for p in paths]
        transform = get_val_transforms(image_size=224)

        ds = CrackClassificationDataset(paths, labels, transform=transform)

        for i in range(len(ds)):
            image, label = ds[i]
            assert image.shape == (3, 224, 224), f"Sample {i}: unexpected shape {image.shape}"
            assert image.dtype == torch.float32
            assert torch.isfinite(image).all(), f"Sample {i}: contains NaN/Inf"
            assert label in (0, 1)

    def test_datamodule_full_cycle(self, real_sdnet2018_dir: Path, tmp_path: Path) -> None:
        """DataModule setup → split → train/val dataloaders produce correct batches."""
        split_path = tmp_path / "cls_split.json"
        splits = create_classification_splits(
            real_sdnet2018_dir, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        split_path.write_text(json.dumps(splits))

        dm = CrackClassificationDM(
            data_dir=real_sdnet2018_dir,
            split_file=split_path,
            batch_size=4,
            num_workers=0,
            image_size=224,
            aug_preset="medium",
        )
        dm.setup("fit")

        # Train dataloader
        images, labels = next(iter(dm.train_dataloader()))
        assert images.ndim == 4  # (N, C, H, W)
        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert set(labels.tolist()).issubset({0, 1})

    @pytest.mark.parametrize("preset", ["light", "medium", "heavy"])
    def test_augmentation_presets_on_real_images(
        self, real_sdnet2018_dir: Path, preset: str
    ) -> None:
        """All augmentation presets produce valid output from real images."""
        transform = get_train_transforms(preset, image_size=224)
        paths = sorted(real_sdnet2018_dir.rglob("*.jpg"))

        for path in paths:
            from PIL import Image

            img = np.array(Image.open(path).convert("RGB"))
            result = transform(image=img)["image"]

            assert result.shape == (3, 224, 224)
            assert result.dtype == torch.float32
            assert torch.isfinite(result).all(), f"{path.name}: NaN/Inf after {preset} aug"


# ---------------------------------------------------------------------------
# Segmentation pipeline — real CrackSeg9k images
# ---------------------------------------------------------------------------


class TestSegmentationPipelineReal:
    """End-to-end segmentation pipeline on real CrackSeg9k images."""

    def test_split_creation(self, real_crackseg9k_dir: Path) -> None:
        """create_segmentation_splits discovers all 3 fixture pairs."""
        splits = create_segmentation_splits(
            real_crackseg9k_dir, val_ratio=0.10, test_ratio=0.10, seed=42
        )
        total = sum(len(v) for v in splits.values())
        assert total == _CRACKSEG9K_N

    def test_dataset_loads_real_pairs(self, real_crackseg9k_dir: Path) -> None:
        """CrackSegmentationDataset returns valid tensors from real PNGs."""
        images_dir = real_crackseg9k_dir / "images"
        masks_dir = real_crackseg9k_dir / "masks" / "Masks"
        names = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
        image_paths = [images_dir / n for n in names]
        mask_paths = [masks_dir / n for n in names]

        transform = get_val_transforms(image_size=224)
        ds = CrackSegmentationDataset(image_paths, mask_paths, transform=transform)

        for i in range(len(ds)):
            image, mask = ds[i]
            assert image.shape == (3, 224, 224), f"Sample {i}: image shape {image.shape}"
            assert image.dtype == torch.float32
            assert torch.isfinite(image).all(), f"Sample {i}: image contains NaN/Inf"

            assert mask.shape == (224, 224), f"Sample {i}: mask shape {mask.shape}"
            assert mask.dtype == torch.long
            unique = set(mask.unique().tolist())
            assert unique.issubset({0, 1}), f"Sample {i}: mask values {unique}, expected {{0, 1}}"

    def test_real_mask_binarization(self, real_crackseg9k_dir: Path) -> None:
        """Real masks with anti-aliased edges are properly binarized to {0, 1}.

        CrackSeg9k masks have intermediate grayscale values along crack
        boundaries. The Dataset thresholds at 127 — verify no values leak through.
        """
        images_dir = real_crackseg9k_dir / "images"
        masks_dir = real_crackseg9k_dir / "masks" / "Masks"
        names = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
        image_paths = [images_dir / n for n in names]
        mask_paths = [masks_dir / n for n in names]

        # Use larger image_size to stress anti-aliasing during resize
        transform = get_val_transforms(image_size=512)
        ds = CrackSegmentationDataset(image_paths, mask_paths, transform=transform)

        for i in range(len(ds)):
            _, mask = ds[i]
            unique = set(mask.unique().tolist())
            assert unique.issubset({0, 1}), (
                f"Sample {i}: anti-aliased mask not properly binarized, got {unique}"
            )

    def test_datamodule_full_cycle(self, real_crackseg9k_dir: Path, tmp_path: Path) -> None:
        """DataModule setup → auto-split → train dataloader produces correct batches."""
        split_path = tmp_path / "seg_split.json"

        dm = CrackSegmentationDM(
            data_dir=real_crackseg9k_dir,
            split_file=split_path,
            batch_size=2,
            num_workers=0,
            image_size=224,
            aug_preset="medium",
        )
        dm.setup("fit")

        # Split file was auto-generated
        assert split_path.exists()

        images, masks = next(iter(dm.train_dataloader()))
        assert images.ndim == 4
        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert masks.ndim == 3
        assert masks.shape[1:] == (224, 224)
        assert masks.dtype == torch.long

    @pytest.mark.parametrize("preset", ["light", "medium", "heavy"])
    def test_augmentation_preserves_mask_binary(
        self, real_crackseg9k_dir: Path, preset: str
    ) -> None:
        """Spatial augmentations on real images keep masks binary {0, 1}."""
        images_dir = real_crackseg9k_dir / "images"
        masks_dir = real_crackseg9k_dir / "masks" / "Masks"
        names = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")

        transform = get_train_transforms(preset, image_size=224)

        for name in names:
            from PIL import Image

            img = np.array(Image.open(images_dir / name).convert("RGB"))
            msk = np.array(Image.open(masks_dir / name).convert("L"))
            # Pre-binarize like the Dataset does (threshold at 127)
            msk = (msk > 127).astype(np.uint8)

            # Run multiple random trials per image
            for trial in range(10):
                result = transform(image=img, mask=msk)
                mask_out = result["mask"]
                if isinstance(mask_out, torch.Tensor):
                    unique = set(mask_out.unique().tolist())
                else:
                    unique = set(np.unique(mask_out).tolist())
                assert unique.issubset({0, 1}), (
                    f"{name} trial {trial} ({preset}): mask values {unique}"
                )
