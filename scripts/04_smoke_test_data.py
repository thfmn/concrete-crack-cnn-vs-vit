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

"""End-to-end smoke test for both data pipelines.

Loads one batch from each DataModule (classification + segmentation),
verifies shapes/dtypes/value ranges, and prints timing statistics.
This is the final gate before training — run it after any pipeline change.

Usage:
    uv run python scripts/04_smoke_test_data.py
"""

from __future__ import annotations

import sys
import time

import torch

from src.data.classification_dm import CrackClassificationDM
from src.data.segmentation_dm import CrackSegmentationDM

# Expected ranges after ImageNet normalization:
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# Theoretical range: (0 - 0.485) / 0.229 ≈ -2.12  to  (1 - 0.406) / 0.225 ≈ 2.64
_IMG_MIN = -2.5
_IMG_MAX = 2.7


def _check_classification(batch_size: int = 32) -> None:
    """Verify classification pipeline shapes, dtypes, and value ranges."""
    print("=" * 60)
    print("CLASSIFICATION PIPELINE (SDNET2018)")
    print("=" * 60)

    dm = CrackClassificationDM(
        data_dir="assets/sdnet2018",
        split_file="configs/splits/sdnet2018_split.json",
        batch_size=batch_size,
        num_workers=4,
        image_size=224,
        aug_preset="medium",
    )
    dm.setup("fit")

    loader = dm.train_dataloader()
    print(f"  Train dataset size: {len(loader.dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {len(loader)}")

    # --- 1-batch timing ---
    t0 = time.perf_counter()
    images, labels = next(iter(loader))
    t1 = time.perf_counter()
    print(f"\n  1-batch load time: {t1 - t0:.3f}s")

    # Shape assertions
    assert images.shape == (batch_size, 3, 224, 224), (
        f"Expected ({batch_size}, 3, 224, 224), got {images.shape}"
    )
    assert labels.shape == (batch_size,), f"Expected ({batch_size},), got {labels.shape}"
    print(f"  Image shape: {images.shape}  ✓")
    print(f"  Label shape: {labels.shape}  ✓")

    # Dtype assertions
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert labels.dtype == torch.int64, f"Expected int64, got {labels.dtype}"
    print(f"  Image dtype: {images.dtype}  ✓")
    print(f"  Label dtype: {labels.dtype}  ✓")

    # Value range assertions
    assert images.min() >= _IMG_MIN, f"Image min {images.min():.3f} < {_IMG_MIN}"
    assert images.max() <= _IMG_MAX, f"Image max {images.max():.3f} > {_IMG_MAX}"
    unique_labels = labels.unique().sort().values
    assert torch.equal(unique_labels, torch.tensor([0, 1])) or all(
        val in (0, 1) for val in unique_labels.tolist()
    ), f"Labels should be {{0, 1}}, got {unique_labels.tolist()}"
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]  ✓")
    print(f"  Label unique values: {unique_labels.tolist()}  ✓")

    # Stats
    print(f"\n  Image mean (per-channel): {images.mean(dim=(0, 2, 3)).tolist()}")
    print(f"  Image std  (per-channel): {images.std(dim=(0, 2, 3)).tolist()}")

    # --- 10-batch timing ---
    t0 = time.perf_counter()
    for i, (_imgs, _lbls) in enumerate(loader):
        if i >= 9:
            break
    t10 = time.perf_counter()
    print(f"\n  10-batch load time: {t10 - t0:.3f}s ({(t10 - t0) / 10:.3f}s/batch)")

    print("\n  CLASSIFICATION SMOKE TEST PASSED ✓")


def _check_segmentation(batch_size: int = 8) -> None:
    """Verify segmentation pipeline shapes, dtypes, and value ranges."""
    print()
    print("=" * 60)
    print("SEGMENTATION PIPELINE (CrackSeg9k)")
    print("=" * 60)

    dm = CrackSegmentationDM(
        data_dir="assets/crackseg9k",
        split_file="configs/splits/crackseg9k_split.json",
        batch_size=batch_size,
        num_workers=4,
        image_size=224,
        aug_preset="medium",
    )
    dm.setup("fit")

    loader = dm.train_dataloader()
    print(f"  Train dataset size: {len(loader.dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {len(loader)}")

    # --- 1-batch timing ---
    t0 = time.perf_counter()
    images, masks = next(iter(loader))
    t1 = time.perf_counter()
    print(f"\n  1-batch load time: {t1 - t0:.3f}s")

    # Shape assertions
    assert images.shape == (batch_size, 3, 224, 224), (
        f"Expected ({batch_size}, 3, 224, 224), got {images.shape}"
    )
    assert masks.shape == (batch_size, 224, 224), (
        f"Expected ({batch_size}, 224, 224), got {masks.shape}"
    )
    print(f"  Image shape: {images.shape}  ✓")
    print(f"  Mask shape:  {masks.shape}  ✓")

    # Dtype assertions
    assert images.dtype == torch.float32, f"Expected float32, got {images.dtype}"
    assert masks.dtype == torch.int64, f"Expected int64, got {masks.dtype}"
    print(f"  Image dtype: {images.dtype}  ✓")
    print(f"  Mask dtype:  {masks.dtype}  ✓")

    # Value range assertions
    assert images.min() >= _IMG_MIN, f"Image min {images.min():.3f} < {_IMG_MIN}"
    assert images.max() <= _IMG_MAX, f"Image max {images.max():.3f} > {_IMG_MAX}"
    unique_mask_vals = masks.unique().sort().values
    assert all(v in (0, 1) for v in unique_mask_vals.tolist()), (
        f"Mask values should be {{0, 1}}, got {unique_mask_vals.tolist()}"
    )
    print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]  ✓")
    print(f"  Mask unique values: {unique_mask_vals.tolist()}  ✓")

    # Stats
    print(f"\n  Image mean (per-channel): {images.mean(dim=(0, 2, 3)).tolist()}")
    print(f"  Image std  (per-channel): {images.std(dim=(0, 2, 3)).tolist()}")
    crack_ratio = masks.float().mean().item()
    print(f"  Batch crack pixel ratio: {crack_ratio:.4f} ({crack_ratio * 100:.2f}%)")

    # --- 10-batch timing ---
    t0 = time.perf_counter()
    for i, (_imgs, _msks) in enumerate(loader):
        if i >= 9:
            break
    t10 = time.perf_counter()
    print(f"\n  10-batch load time: {t10 - t0:.3f}s ({(t10 - t0) / 10:.3f}s/batch)")

    print("\n  SEGMENTATION SMOKE TEST PASSED ✓")


def main() -> None:
    """Run smoke tests for both pipelines."""
    print()
    print("DATA-11: End-to-End Data Pipeline Smoke Test")
    print()

    failed = False
    try:
        _check_classification(batch_size=32)
    except Exception as e:
        print(f"\n  CLASSIFICATION SMOKE TEST FAILED: {e}")
        failed = True

    try:
        _check_segmentation(batch_size=8)
    except Exception as e:
        print(f"\n  SEGMENTATION SMOKE TEST FAILED: {e}")
        failed = True

    print()
    if failed:
        print("SMOKE TEST RESULT: FAILED")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED ✓")
        print("Data pipeline is ready for training.")


if __name__ == "__main__":
    main()
