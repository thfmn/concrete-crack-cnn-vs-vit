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

"""Unit tests for the CrackSeg9k segmentation data pipeline.

Tests cover:
- CrackSegmentationDataset: shapes, dtypes, mask binarization
- CrackSegmentationDM: setup, split sizes, dataloader output
- Spatial augmentation consistency: image and mask transform identically
- Segmentation splits: sizes, no overlap, determinism

All tests use synthetic dummy images + masks in a tmp_path fixture —
no dependency on the real downloaded dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.segmentation_dataset import CrackSegmentationDataset
from src.data.segmentation_dm import CrackSegmentationDM, create_segmentation_splits

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NUM_IMAGES = 20
_IMAGE_SIZE = 64  # small for speed


@pytest.fixture()
def synthetic_seg_dataset(tmp_path: Path) -> Path:
    """Create a miniature CrackSeg9k directory with random images and binary masks.

    Layout mirrors real data:
        tmp_path/images/*.png       (RGB)
        tmp_path/masks/Masks/*.png  (grayscale, 0 or 255)
    """
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks" / "Masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    for i in range(_NUM_IMAGES):
        # Random RGB image
        pixels = rng.integers(0, 256, (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(images_dir / f"img_{i:04d}.png")

        # Binary mask: random 0/255 values (simulates crack vs background)
        mask_vals = rng.choice([0, 255], size=(_IMAGE_SIZE, _IMAGE_SIZE)).astype(np.uint8)
        Image.fromarray(mask_vals, mode="L").save(masks_dir / f"img_{i:04d}.png")

    return tmp_path


@pytest.fixture()
def split_file(synthetic_seg_dataset: Path, tmp_path: Path) -> Path:
    """Generate a segmentation split JSON from the synthetic dataset."""
    splits = create_segmentation_splits(
        synthetic_seg_dataset, val_ratio=0.10, test_ratio=0.10, seed=42
    )
    out = tmp_path / "seg_split.json"
    out.write_text(json.dumps(splits))
    return out


def _get_image_and_mask_paths(
    data_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Helper: list paired image and mask paths from the synthetic directory."""
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks" / "Masks"
    names = sorted(f.name for f in images_dir.iterdir() if f.suffix == ".png")
    image_paths = [images_dir / n for n in names]
    mask_paths = [masks_dir / n for n in names]
    return image_paths, mask_paths


# ---------------------------------------------------------------------------
# CrackSegmentationDataset tests
# ---------------------------------------------------------------------------


class TestCrackSegmentationDataset:
    """Tests for the per-sample Dataset (image+mask loader)."""

    def test_returns_correct_shapes_and_dtypes(self, synthetic_seg_dataset: Path) -> None:
        """__getitem__ returns (image [3,224,224] float32, mask [224,224] int64)."""
        image_paths, mask_paths = _get_image_and_mask_paths(synthetic_seg_dataset)
        transform = get_val_transforms(image_size=224)

        ds = CrackSegmentationDataset(image_paths, mask_paths, transform=transform)
        image, mask = ds[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32

        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (224, 224)
        assert mask.dtype == torch.long

    def test_mask_values_are_binary(self, synthetic_seg_dataset: Path) -> None:
        """After thresholding, mask contains only 0 and 1."""
        image_paths, mask_paths = _get_image_and_mask_paths(synthetic_seg_dataset)
        transform = get_val_transforms(image_size=224)

        ds = CrackSegmentationDataset(image_paths, mask_paths, transform=transform)

        for i in range(min(5, len(ds))):
            _, mask = ds[i]
            unique_vals = set(mask.unique().tolist())
            assert unique_vals.issubset({0, 1}), (
                f"Sample {i}: mask contains values {unique_vals}, expected {{0, 1}}"
            )

    def test_length_matches_inputs(self, synthetic_seg_dataset: Path) -> None:
        image_paths, mask_paths = _get_image_and_mask_paths(synthetic_seg_dataset)
        ds = CrackSegmentationDataset(image_paths, mask_paths)
        assert len(ds) == _NUM_IMAGES

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            CrackSegmentationDataset([Path("a.png")], [Path("a.png"), Path("b.png")])

    def test_antialiased_mask_is_binarized(self, tmp_path: Path) -> None:
        """Masks with intermediate values (anti-aliased edges) are thresholded at 127."""
        images_dir = tmp_path / "aa_images"
        masks_dir = tmp_path / "aa_masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        # Create one image + one mask with anti-aliased values (0, 64, 128, 192, 255)
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        Image.fromarray(img).save(images_dir / "test.png")

        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[0:16, :] = 0  # below threshold → 0
        mask_arr[16:32, :] = 64  # below threshold → 0
        mask_arr[32:48, :] = 128  # at threshold → 1 (>127)
        mask_arr[48:64, :] = 255  # above threshold → 1
        Image.fromarray(mask_arr, mode="L").save(masks_dir / "test.png")

        transform = get_val_transforms(image_size=64)
        ds = CrackSegmentationDataset(
            [images_dir / "test.png"],
            [masks_dir / "test.png"],
            transform=transform,
        )
        _, mask = ds[0]

        assert set(mask.unique().tolist()) == {0, 1}
        # Top half (rows 0–31) should be 0, bottom half (rows 32–63) should be 1
        assert mask[:32, :].sum() == 0
        assert (mask[32:, :] == 1).all()


# ---------------------------------------------------------------------------
# Spatial augmentation consistency tests
# ---------------------------------------------------------------------------


class TestSpatialAugmentationConsistency:
    """Verify spatial transforms are applied identically to image and mask."""

    def test_horizontal_flip_consistency(self) -> None:
        """When HorizontalFlip fires (p=1.0), both image and mask are flipped."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = A.Compose(
            [
                A.Resize(64, 64),
                A.HorizontalFlip(p=1.0),  # always flip
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        rng = np.random.default_rng(99)
        image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        # Asymmetric mask: left half = 1, right half = 0
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[:, :32] = 1

        result = transform(image=image, mask=mask)
        flipped_mask = result["mask"]

        # After flip: left half should be 0, right half should be 1
        assert (flipped_mask[:, :32] == 0).all()
        assert (flipped_mask[:, 32:] == 1).all()

    def test_spatial_transforms_preserve_mask_values(self) -> None:
        """Spatial transforms (resize, flip, affine) don't introduce new mask values.

        Uses nearest-neighbor interpolation (albumentations default for masks)
        so binary masks stay binary.
        """
        transform = get_train_transforms("medium", image_size=224)

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = rng.choice([0, 1], size=(100, 100)).astype(np.uint8)

        # Run 20 random trials — mask should always stay {0, 1}
        for _ in range(20):
            result = transform(image=image, mask=mask)
            mask_out = result["mask"]
            unique = (
                set(np.unique(mask_out).tolist())
                if isinstance(mask_out, np.ndarray)
                else set(torch.unique(torch.as_tensor(mask_out)).tolist())
            )
            assert unique.issubset({0, 1}), f"Mask contains unexpected values: {unique}"

    @pytest.mark.parametrize("preset", ["light", "medium", "heavy"])
    def test_mask_shape_matches_image(self, preset: str) -> None:
        """After any preset transform, mask spatial dims match image spatial dims."""
        transform = get_train_transforms(preset, image_size=224)

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = rng.choice([0, 1], size=(100, 100)).astype(np.uint8)

        result = transform(image=image, mask=mask)
        img_out = result["image"]
        mask_out = result["mask"]

        # image is [C, H, W] tensor; mask is [H, W] numpy or tensor
        assert img_out.shape[1:] == (224, 224)
        if isinstance(mask_out, torch.Tensor):
            assert mask_out.shape == (224, 224)
        else:
            assert mask_out.shape == (224, 224)


# ---------------------------------------------------------------------------
# CrackSegmentationDM tests
# ---------------------------------------------------------------------------


class TestCrackSegmentationDM:
    """Tests for the Lightning DataModule."""

    def test_setup_creates_correct_split_sizes(
        self, synthetic_seg_dataset: Path, split_file: Path
    ) -> None:
        """After setup('fit'), train and val datasets have expected sizes."""
        dm = CrackSegmentationDM(
            data_dir=synthetic_seg_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=224,
        )
        dm.setup("fit")

        with split_file.open() as f:
            splits = json.load(f)

        assert len(dm._train_dataset) == len(splits["train"])
        assert len(dm._val_dataset) == len(splits["val"])

    def test_setup_test_stage(self, synthetic_seg_dataset: Path, split_file: Path) -> None:
        dm = CrackSegmentationDM(
            data_dir=synthetic_seg_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("test")

        with split_file.open() as f:
            splits = json.load(f)

        assert len(dm._test_dataset) == len(splits["test"])

    def test_train_dataloader_batch_shapes(
        self, synthetic_seg_dataset: Path, split_file: Path
    ) -> None:
        """One batch from train_dataloader has correct shapes and dtypes."""
        dm = CrackSegmentationDM(
            data_dir=synthetic_seg_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=224,
        )
        dm.setup("fit")
        images, masks = next(iter(dm.train_dataloader()))

        assert images.shape == (4, 3, 224, 224)
        assert images.dtype == torch.float32
        assert masks.shape == (4, 224, 224)
        assert masks.dtype == torch.long
        # All mask values should be 0 or 1
        assert set(masks.unique().tolist()).issubset({0, 1})

    def test_val_dataloader_batch_shapes(
        self, synthetic_seg_dataset: Path, split_file: Path
    ) -> None:
        dm = CrackSegmentationDM(
            data_dir=synthetic_seg_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=224,
        )
        dm.setup("fit")
        images, masks = next(iter(dm.val_dataloader()))

        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert masks.shape[1:] == (224, 224)
        assert masks.dtype == torch.long

    def test_generates_split_file_if_missing(
        self, synthetic_seg_dataset: Path, tmp_path: Path
    ) -> None:
        """If split_file doesn't exist, setup() creates it automatically."""
        new_split = tmp_path / "auto_split.json"
        assert not new_split.exists()

        dm = CrackSegmentationDM(
            data_dir=synthetic_seg_dataset,
            split_file=new_split,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("fit")

        assert new_split.exists()
        splits = json.loads(new_split.read_text())
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == _NUM_IMAGES


# ---------------------------------------------------------------------------
# Segmentation split tests
# ---------------------------------------------------------------------------


class TestSegmentationSplits:
    """Tests for create_segmentation_splits."""

    def test_split_sizes(self, synthetic_seg_dataset: Path) -> None:
        """80/10/10 split of 20 images: 16/2/2."""
        splits = create_segmentation_splits(
            synthetic_seg_dataset, val_ratio=0.10, test_ratio=0.10, seed=42
        )
        total = sum(len(v) for v in splits.values())
        assert total == _NUM_IMAGES

    def test_no_overlap_between_splits(self, synthetic_seg_dataset: Path) -> None:
        splits = create_segmentation_splits(synthetic_seg_dataset, seed=42)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_deterministic_with_same_seed(self, synthetic_seg_dataset: Path) -> None:
        splits_a = create_segmentation_splits(synthetic_seg_dataset, seed=42)
        splits_b = create_segmentation_splits(synthetic_seg_dataset, seed=42)

        assert splits_a == splits_b
