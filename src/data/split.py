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

"""Stratified train/val/test splitting for SDNET2018 classification.

Surface types (D=Decks, P=Pavements, W=Walls) are merged into a single pool.
This creates a harder, more generalizable benchmark — the model must learn to
detect cracks regardless of surface texture. Keeping surfaces separate would
let each sub-model overfit to one texture and inflate reported accuracy.

Equivalent in Keras/TF:
    tf.keras.utils.image_dataset_from_directory(..., validation_split=0.15, subset="training")
    only supports a single train/val split. This module creates a proper train/val/test split
    with stratification, which Keras doesn't natively support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Labels: 1 = cracked, 0 = uncracked
_CRACKED_PREFIXES = ("CD", "CP", "CW")
_UNCRACKED_PREFIXES = ("UD", "UP", "UW")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_image_paths(data_dir: Path) -> tuple[list[str], list[int]]:
    """Walk the SDNET2018 directory tree and collect (relative_path, label) pairs.

    Returns relative paths (e.g. "D/CD/7001-115.jpg") so the split file stays
    portable across machines — just prepend the local data_dir at load time.
    """
    paths: list[str] = []
    labels: list[int] = []

    for surface_dir in sorted(data_dir.iterdir()):
        if not surface_dir.is_dir():
            continue
        for class_dir in sorted(surface_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            dir_name = class_dir.name
            if dir_name in _CRACKED_PREFIXES:
                label = 1
            elif dir_name in _UNCRACKED_PREFIXES:
                label = 0
            else:
                logger.warning("Skipping unknown directory: %s", class_dir)
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    # Store path relative to data_dir
                    paths.append(str(img_path.relative_to(data_dir)))
                    labels.append(label)

    return paths, labels


def create_classification_splits(
    data_dir: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Create stratified train/val/test splits for SDNET2018.

    All surface types (D, P, W) are merged into one pool. The split preserves
    class balance (cracked vs uncracked ratio) in each partition.

    Equivalent in Keras/TF:
        There's no direct equivalent — tf.keras.utils.image_dataset_from_directory
        only supports a single train/val split. Here we use scikit-learn's
        train_test_split with stratify= for a proper 3-way stratified split.

    Args:
        data_dir: Path to assets/sdnet2018/ containing D/, P/, W/ subdirectories.
        val_ratio: Fraction of data for validation (default 0.15).
        test_ratio: Fraction of data for test (default 0.15).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Dict with "train", "val", "test" keys mapping to lists of relative paths.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        msg = f"Data directory does not exist: {data_dir}"
        raise FileNotFoundError(msg)

    paths, labels = _collect_image_paths(data_dir)
    if len(paths) == 0:
        msg = f"No images found in {data_dir}"
        raise ValueError(msg)

    n_cracked = sum(labels)
    logger.info(
        "Collected %d images (%d cracked, %d uncracked)",
        len(paths),
        n_cracked,
        len(labels) - n_cracked,
    )

    # Two-step split: first separate out test, then split remainder into train/val.
    # This is the standard approach since sklearn doesn't have a 3-way stratified split.
    val_test_ratio = val_ratio + test_ratio
    paths_train, paths_valtest, labels_train, labels_valtest = train_test_split(
        paths,
        labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Split the val+test portion into val and test
    # test_ratio / (val_ratio + test_ratio) gives the fraction of the held-out set for test
    test_fraction_of_valtest = test_ratio / val_test_ratio
    paths_val, paths_test, _, _ = train_test_split(
        paths_valtest,
        labels_valtest,
        test_size=test_fraction_of_valtest,
        stratify=labels_valtest,
        random_state=seed,
    )

    splits = {
        "train": sorted(paths_train),
        "val": sorted(paths_val),
        "test": sorted(paths_test),
    }

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    return splits


def save_splits(
    splits: dict[str, list[str]],
    output_path: str | Path,
) -> Path:
    """Save split dictionary to a JSON file.

    Args:
        splits: Dict from create_classification_splits().
        output_path: Where to write the JSON file.

    Returns:
        The resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(splits, f, indent=2)
    logger.info("Saved splits to %s", output_path)
    return output_path


if __name__ == "__main__":
    # Convenience: generate the split file when run directly
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "assets" / "sdnet2018"
    output_path = project_root / "configs" / "splits" / "sdnet2018_split.json"

    if not data_dir.exists():
        logger.error("Dataset not found at %s", data_dir)
        sys.exit(1)

    splits = create_classification_splits(data_dir)
    save_splits(splits, output_path)

    # Print class balance per split for verification
    for split_name, file_paths in splits.items():
        n_cracked = sum(1 for p in file_paths if Path(p).parent.name in _CRACKED_PREFIXES)
        n_uncracked = len(file_paths) - n_cracked
        ratio = n_cracked / n_uncracked if n_uncracked > 0 else float("inf")
        print(
            f"  {split_name:5s}: {len(file_paths):6d} images "
            f"({n_cracked} cracked / {n_uncracked} uncracked, ratio={ratio:.4f})"
        )
