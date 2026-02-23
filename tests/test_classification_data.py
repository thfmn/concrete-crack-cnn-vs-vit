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

"""Unit tests for the SDNET2018 classification data pipeline.

Tests cover:
- CrackClassificationDataset: shapes, dtypes, transform application
- CrackClassificationDM: setup, split sizes, dataloader output
- Augmentation transforms: output shapes, val/test determinism
- Stratified splits: class balance preservation

All tests use synthetic dummy images in a tmp_path fixture —
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
from src.data.classification_dataset import CrackClassificationDataset
from src.data.classification_dm import CrackClassificationDM
from src.data.split import create_classification_splits

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Synthetic dataset: 5 cracked + 5 uncracked per surface type = 30 images total.
# Mirrors the SDNET2018 layout: D/{CD,UD}, P/{CP,UP}, W/{CW,UW}.
_SURFACE_CLASSES = [
    ("D", "CD", 1),
    ("D", "UD", 0),
    ("P", "CP", 1),
    ("P", "UP", 0),
    ("W", "CW", 1),
    ("W", "UW", 0),
]
_IMAGES_PER_CLASS = 5
_IMAGE_SIZE = 64  # small for speed


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a miniature SDNET2018 directory with random PNG images."""
    rng = np.random.default_rng(42)
    for surface, class_dir, _label in _SURFACE_CLASSES:
        dir_path = tmp_path / surface / class_dir
        dir_path.mkdir(parents=True)
        for i in range(_IMAGES_PER_CLASS):
            pixels = rng.integers(0, 256, (_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(dir_path / f"{i:04d}.png")
    return tmp_path


@pytest.fixture()
def split_file(synthetic_dataset: Path, tmp_path: Path) -> Path:
    """Generate a split JSON from the synthetic dataset."""
    splits = create_classification_splits(
        synthetic_dataset, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    out = tmp_path / "split.json"
    out.write_text(json.dumps(splits))
    return out


# ---------------------------------------------------------------------------
# CrackClassificationDataset tests
# ---------------------------------------------------------------------------


class TestCrackClassificationDataset:
    """Tests for the Dataset class (the per-sample loader)."""

    def test_returns_correct_shapes_and_dtypes(self, synthetic_dataset: Path) -> None:
        """Dataset __getitem__ returns (image [3,224,224] float32, label int)."""
        paths = list(synthetic_dataset.rglob("*.png"))
        labels = [1 if p.parent.name.startswith("C") else 0 for p in paths]
        transform = get_val_transforms(image_size=224)

        ds = CrackClassificationDataset(paths, labels, transform=transform)
        image, label = ds[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert label in (0, 1)

    def test_length_matches_inputs(self, synthetic_dataset: Path) -> None:
        paths = list(synthetic_dataset.rglob("*.png"))
        labels = [0] * len(paths)
        ds = CrackClassificationDataset(paths, labels)
        assert len(ds) == len(paths)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            CrackClassificationDataset([Path("a.png")], [0, 1])

    def test_all_labels_returned_correctly(self, synthetic_dataset: Path) -> None:
        """Every sample's label matches the directory-based ground truth."""
        paths = sorted(synthetic_dataset.rglob("*.png"))
        labels = [1 if p.parent.name.startswith("C") else 0 for p in paths]
        transform = get_val_transforms(image_size=224)
        ds = CrackClassificationDataset(paths, labels, transform=transform)

        for i in range(len(ds)):
            _, returned_label = ds[i]
            assert returned_label == labels[i]


# ---------------------------------------------------------------------------
# CrackClassificationDM tests
# ---------------------------------------------------------------------------


class TestCrackClassificationDM:
    """Tests for the Lightning DataModule."""

    def test_setup_creates_correct_split_sizes(
        self, synthetic_dataset: Path, split_file: Path
    ) -> None:
        """After setup('fit'), train and val datasets have the expected sizes."""
        dm = CrackClassificationDM(
            data_dir=synthetic_dataset,
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

    def test_setup_test_stage(self, synthetic_dataset: Path, split_file: Path) -> None:
        dm = CrackClassificationDM(
            data_dir=synthetic_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
        )
        dm.setup("test")

        with split_file.open() as f:
            splits = json.load(f)

        assert len(dm._test_dataset) == len(splits["test"])

    def test_train_dataloader_batch_shapes(self, synthetic_dataset: Path, split_file: Path) -> None:
        """One batch from train_dataloader has correct shapes and dtypes."""
        dm = CrackClassificationDM(
            data_dir=synthetic_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=224,
        )
        dm.setup("fit")
        images, labels = next(iter(dm.train_dataloader()))

        assert images.shape == (4, 3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.shape == (4,)
        assert labels.dtype == torch.int64
        assert set(labels.tolist()).issubset({0, 1})

    def test_val_dataloader_batch_shapes(self, synthetic_dataset: Path, split_file: Path) -> None:
        dm = CrackClassificationDM(
            data_dir=synthetic_dataset,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=224,
        )
        dm.setup("fit")
        images, labels = next(iter(dm.val_dataloader()))

        assert images.shape[1:] == (3, 224, 224)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64


# ---------------------------------------------------------------------------
# Augmentation transform tests
# ---------------------------------------------------------------------------


class TestAugmentationTransforms:
    """Tests for get_train_transforms and get_val_transforms."""

    @pytest.mark.parametrize("preset", ["light", "medium", "heavy"])
    def test_train_transform_output_shape(self, preset: str) -> None:
        """All presets produce (3, 224, 224) float32 tensor from a 256x256 image."""
        transform = get_train_transforms(preset, image_size=224)
        dummy = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        result = transform(image=dummy)["image"]

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_val_transform_is_deterministic(self) -> None:
        """Val transform applied twice to the same image produces identical output."""
        transform = get_val_transforms(image_size=224)
        dummy = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)

        result_a = transform(image=dummy)["image"]
        result_b = transform(image=dummy)["image"]

        assert torch.equal(result_a, result_b)

    def test_val_transform_output_range(self) -> None:
        """After ImageNet normalization, values are roughly in [-2.5, 2.7]."""
        transform = get_val_transforms(image_size=224)
        dummy = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        result = transform(image=dummy)["image"]

        assert result.min() >= -3.0
        assert result.max() <= 3.0


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------


class TestStratifiedSplits:
    """Tests for create_classification_splits."""

    def test_split_sizes(self, synthetic_dataset: Path) -> None:
        """70/15/15 split of 30 images: 21/4-5/4-5 (rounding)."""
        splits = create_classification_splits(
            synthetic_dataset, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        total = sum(len(v) for v in splits.values())
        assert total == 30  # 5 images × 6 class dirs

    def test_no_overlap_between_splits(self, synthetic_dataset: Path) -> None:
        splits = create_classification_splits(synthetic_dataset, seed=42)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_deterministic_with_same_seed(self, synthetic_dataset: Path) -> None:
        splits_a = create_classification_splits(synthetic_dataset, seed=42)
        splits_b = create_classification_splits(synthetic_dataset, seed=42)

        assert splits_a == splits_b

    def test_class_balance_preserved(self, synthetic_dataset: Path) -> None:
        """Cracked ratio should be similar across all splits (within ±10% for small N)."""
        splits = create_classification_splits(
            synthetic_dataset, val_ratio=0.15, test_ratio=0.15, seed=42
        )

        def cracked_ratio(paths: list[str]) -> float:
            n_cracked = sum(1 for p in paths if Path(p).parent.name.startswith("C"))
            return n_cracked / len(paths) if paths else 0.0

        overall = cracked_ratio(splits["train"] + splits["val"] + splits["test"])

        for split_name in ("train", "val", "test"):
            ratio = cracked_ratio(splits[split_name])
            # With only 30 images, allow ±10% tolerance
            assert abs(ratio - overall) < 0.10, (
                f"{split_name} cracked ratio {ratio:.2f} deviates from "
                f"overall {overall:.2f} by more than 10%"
            )
