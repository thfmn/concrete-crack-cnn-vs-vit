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
#  Package:   concrete-crack-cnn-vs-vit -- CNN vs ViT Benchmark

"""Evaluation script for trained classification and segmentation models.

Loads checkpoint(s), runs inference on the test set, computes metrics,
and saves comparison results as JSON and CSV files.

Usage:
    uv run python scripts/evaluate.py                          # all checkpoints
    uv run python scripts/evaluate.py --checkpoint path.ckpt   # single checkpoint
    uv run python scripts/evaluate.py --task classification     # only cls checkpoints
    uv run python scripts/evaluate.py --task segmentation       # only seg checkpoints
    uv run python scripts/evaluate.py --batch-size 64
    uv run python scripts/evaluate.py --output-dir outputs/results/evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from tqdm import tqdm

from src.data.classification_dm import CrackClassificationDM
from src.data.segmentation_dm import CrackSegmentationDM
from src.models.classification_module import CrackClassifier
from src.models.segmentation_module import CrackSegmentor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model name -> architecture family mapping.
# Covers both classification (timm_name) and segmentation (encoder_name) keys.
# ---------------------------------------------------------------------------
_FAMILY_MAP: dict[str, str] = {
    # Classification models (timm keys)
    "resnet50": "cnn",
    "tf_efficientnetv2_s": "cnn",
    "convnext_tiny": "cnn",
    "vit_base_patch16_224": "vit",
    "swin_tiny_patch4_window7_224": "vit",
    "deit_small_patch16_224": "vit",
    # Segmentation encoders (smp/timm-universal keys)
    "tu-tf_efficientnetv2_s": "cnn",
    "tu-convnext_tiny": "cnn",
    "tu-swin_tiny_patch4_window7_224": "vit",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN and ViT crack detection models.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a single checkpoint file (.ckpt). If omitted, all "
        "checkpoints in --checkpoints-dir are evaluated.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory to scan for .ckpt files (default: outputs/checkpoints).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        default=None,
        help="Filter checkpoints by task type. If omitted, both tasks are evaluated.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: 'auto', 'cpu', or 'cuda' (default: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/evaluation"),
        help="Directory to write result files (default: outputs/results/evaluation).",
    )
    return parser.parse_args()


def detect_device(requested: str) -> torch.device:
    """Auto-detect the best available device, or use the one requested.

    Args:
        requested: One of 'auto', 'cpu', or 'cuda'.

    Returns:
        A torch.device pointing to the selected hardware.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Auto-detected CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            logger.info("No CUDA device found, using CPU.")
        return device
    return torch.device(requested)


def detect_task(ckpt_path: Path) -> str:
    """Determine whether a checkpoint is classification or segmentation.

    Loads the checkpoint's hyperparameters and checks for the presence of
    ``encoder_name`` (segmentation) vs ``model_name`` (classification).

    Args:
        ckpt_path: Path to a Lightning .ckpt file.

    Returns:
        "classification" or "segmentation".

    Raises:
        ValueError: If the checkpoint hparams cannot identify the task.
    """
    # Load only the hyper_parameters from the checkpoint, not the full model.
    # map_location="cpu" avoids requiring a GPU just for inspection.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    if "encoder_name" in hparams:
        return "segmentation"
    if "model_name" in hparams:
        return "classification"

    raise ValueError(
        f"Cannot detect task for checkpoint {ckpt_path}. "
        f"Found hparams keys: {sorted(hparams.keys())}"
    )


def discover_checkpoints(
    checkpoints_dir: Path,
    task_filter: str | None,
) -> list[tuple[Path, str]]:
    """Discover checkpoint files and pair each with its detected task.

    Args:
        checkpoints_dir: Directory to glob for *.ckpt files.
        task_filter: If set, only return checkpoints matching this task
            ("classification" or "segmentation").

    Returns:
        A sorted list of (checkpoint_path, task_string) tuples.
    """
    ckpt_paths = sorted(checkpoints_dir.glob("*.ckpt"))
    if not ckpt_paths:
        logger.warning("No .ckpt files found in %s", checkpoints_dir)
        return []

    results: list[tuple[Path, str]] = []
    for path in ckpt_paths:
        try:
            task = detect_task(path)
        except ValueError:
            logger.warning("Skipping unrecognized checkpoint: %s", path)
            continue

        if task_filter is not None and task != task_filter:
            continue
        results.append((path, task))

    logger.info(
        "Discovered %d checkpoint(s) in %s (filter=%s)",
        len(results),
        checkpoints_dir,
        task_filter or "none",
    )
    return results


def extract_model_name(ckpt_path: Path, task: str) -> str:
    """Extract the model/encoder name from a checkpoint's hyperparameters.

    Args:
        ckpt_path: Path to the .ckpt file.
        task: "classification" or "segmentation".

    Returns:
        The model identifier string (e.g. "resnet50", "tu-swin_tiny_patch4_window7_224").
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})

    if task == "segmentation":
        return hparams.get("encoder_name", "unknown")
    return hparams.get("model_name", "unknown")


def resolve_family(model_name: str) -> str:
    """Map a model/encoder name to its architecture family.

    Args:
        model_name: timm or smp encoder key (e.g. "resnet50", "tu-convnext_tiny").

    Returns:
        "cnn" or "vit". Falls back to "unknown" for unrecognized names.
    """
    return _FAMILY_MAP.get(model_name, "unknown")


def evaluate_classification(
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Evaluate a classification checkpoint on the SDNET2018 test set.

    Loads the CrackClassifier from the checkpoint, creates a
    CrackClassificationDM with validation-only transforms, runs inference
    with torch.no_grad(), and computes sklearn metrics.

    Args:
        ckpt_path: Path to the classification .ckpt file.
        device: Device to run inference on.
        batch_size: Batch size for the test DataLoader.
        num_workers: Number of DataLoader workers.

    Returns:
        Dict containing model metadata and all computed metrics.
    """
    # Load model from checkpoint. Lightning restores hparams automatically.
    model = CrackClassifier.load_from_checkpoint(str(ckpt_path), map_location=device)
    # Switch to evaluation mode: disables dropout and batch-norm updates.
    model.eval()
    model.to(device)

    model_name = model.hparams.get("model_name", "unknown")
    family = resolve_family(model_name)

    # Set up test data with validation transforms (no augmentation).
    dm = CrackClassificationDM(
        data_dir="assets/sdnet2018",
        split_file="configs/splits/sdnet2018_split.json",
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        aug_preset="medium",
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []

    # torch.no_grad() disables gradient computation, saving memory and time.
    # This is required for inference; Lightning does it automatically in
    # validation_step, but here we run a manual loop.
    start_time = time.perf_counter()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"  {model_name}", leave=False):
            images = images.to(device)
            logits = model(images)

            # Convert raw logits to class predictions and probabilities.
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = logits.softmax(dim=1)[:, 1].cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    elapsed = time.perf_counter() - start_time

    # Compute sklearn classification metrics on the full test set.
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary"))
    precision = float(precision_score(y_true, y_pred, average="binary"))
    recall_val = float(recall_score(y_true, y_pred, average="binary"))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred, target_names=["no_crack", "crack"], output_dict=True
    )

    return {
        "checkpoint": str(ckpt_path),
        "task": "classification",
        "model_name": model_name,
        "family": family,
        "num_samples": len(y_true),
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall_val,
        "confusion_matrix": cm,
        "classification_report": report,
        "inference_time_s": round(elapsed, 2),
    }


def evaluate_segmentation(
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Evaluate a segmentation checkpoint on the CrackSeg9k test set.

    Loads the CrackSegmentor from the checkpoint, creates a
    CrackSegmentationDM, runs inference with torch.no_grad(), and computes
    IoU (Jaccard) and Dice (F1) using torchmetrics.

    Args:
        ckpt_path: Path to the segmentation .ckpt file.
        device: Device to run inference on.
        batch_size: Batch size for the test DataLoader.
        num_workers: Number of DataLoader workers.

    Returns:
        Dict containing model metadata and all computed metrics.
    """
    model = CrackSegmentor.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)

    encoder_name = model.hparams.get("encoder_name", "unknown")
    family = resolve_family(encoder_name)

    dm = CrackSegmentationDM(
        data_dir="assets/crackseg9k",
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # torchmetrics accumulators for IoU and Dice over the full test set.
    # BinaryJaccardIndex = IoU for binary masks.
    # BinaryF1Score = Dice coefficient for binary masks.
    iou_metric = BinaryJaccardIndex().to(device)
    dice_metric = BinaryF1Score().to(device)

    num_samples = 0

    start_time = time.perf_counter()
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"  {encoder_name}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)

            # Convert logits to binary predictions: sigmoid squashes to [0, 1],
            # then threshold at 0.5.
            probs = logits.sigmoid()
            preds = (probs > 0.5).long()

            # Squeeze channel dimension for metrics (B, 1, H, W) -> (B, H, W).
            preds_flat = preds.squeeze(1)
            masks_flat = masks.squeeze(1).long()

            iou_metric.update(preds_flat, masks_flat)
            dice_metric.update(preds_flat, masks_flat)
            num_samples += images.size(0)

    elapsed = time.perf_counter() - start_time

    iou = float(iou_metric.compute().cpu())
    dice = float(dice_metric.compute().cpu())

    return {
        "checkpoint": str(ckpt_path),
        "task": "segmentation",
        "model_name": encoder_name,
        "family": family,
        "num_samples": num_samples,
        "iou": iou,
        "dice": dice,
        "inference_time_s": round(elapsed, 2),
    }


def save_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Persist evaluation results as JSON and CSV files.

    Creates:
        - all_results.json: full list of result dicts.
        - comparison_classification.csv: classification metrics table.
        - comparison_segmentation.csv: segmentation metrics table.
        - Per-model JSON files in a models/ subdirectory.

    Args:
        results: List of result dicts from evaluate_classification/segmentation.
        output_dir: Directory to write files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- all_results.json ----
    all_results_path = output_dir / "all_results.json"
    all_results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved all results to %s", all_results_path)

    # ---- Per-model JSON files ----
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for result in results:
        model_name = result["model_name"].replace("/", "_")
        model_path = models_dir / f"{model_name}.json"
        model_path.write_text(json.dumps(result, indent=2))

    # ---- Classification CSV ----
    cls_results = [r for r in results if r["task"] == "classification"]
    if cls_results:
        cls_rows = []
        for r in cls_results:
            cls_rows.append(
                {
                    "model": r["model_name"],
                    "family": r["family"],
                    "accuracy": r["accuracy"],
                    "f1": r["f1"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "num_samples": r["num_samples"],
                    "inference_time_s": r["inference_time_s"],
                }
            )
        cls_df = pd.DataFrame(cls_rows)
        cls_csv_path = output_dir / "comparison_classification.csv"
        cls_df.to_csv(cls_csv_path, index=False)
        logger.info("Saved classification comparison to %s", cls_csv_path)

    # ---- Segmentation CSV ----
    seg_results = [r for r in results if r["task"] == "segmentation"]
    if seg_results:
        seg_rows = []
        for r in seg_results:
            seg_rows.append(
                {
                    "model": r["model_name"],
                    "family": r["family"],
                    "iou": r["iou"],
                    "dice": r["dice"],
                    "num_samples": r["num_samples"],
                    "inference_time_s": r["inference_time_s"],
                }
            )
        seg_df = pd.DataFrame(seg_rows)
        seg_csv_path = output_dir / "comparison_segmentation.csv"
        seg_df.to_csv(seg_csv_path, index=False)
        logger.info("Saved segmentation comparison to %s", seg_csv_path)


def _print_summary(results: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table to stdout.

    Args:
        results: List of result dicts from evaluation.
    """
    cls_results = [r for r in results if r["task"] == "classification"]
    seg_results = [r for r in results if r["task"] == "segmentation"]

    if cls_results:
        print("\n" + "=" * 80)
        print("CLASSIFICATION RESULTS")
        print("=" * 80)
        header = f"{'Model':<35} {'Family':<6} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}"
        print(header)
        print("-" * 80)
        for r in cls_results:
            row = (
                f"{r['model_name']:<35} "
                f"{r['family']:<6} "
                f"{r['accuracy']:>7.4f} "
                f"{r['f1']:>7.4f} "
                f"{r['precision']:>7.4f} "
                f"{r['recall']:>7.4f}"
            )
            print(row)
        print()

    if seg_results:
        print("=" * 80)
        print("SEGMENTATION RESULTS")
        print("=" * 80)
        header = f"{'Model':<35} {'Family':<6} {'IoU':>7} {'Dice':>7}"
        print(header)
        print("-" * 80)
        for r in seg_results:
            row = f"{r['model_name']:<35} {r['family']:<6} {r['iou']:>7.4f} {r['dice']:>7.4f}"
            print(row)
        print()

    if not cls_results and not seg_results:
        print("\nNo results to display.")


def main() -> None:
    """Orchestrate checkpoint discovery, evaluation, and result saving."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    device = detect_device(args.device)

    # Build the list of checkpoints to evaluate.
    if args.checkpoint is not None:
        # Single checkpoint mode.
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        task = detect_task(args.checkpoint)
        if args.task is not None and task != args.task:
            raise ValueError(
                f"Checkpoint task ({task}) does not match --task filter ({args.task})."
            )
        checkpoints = [(args.checkpoint, task)]
    else:
        # Discovery mode: scan the checkpoints directory.
        checkpoints = discover_checkpoints(args.checkpoints_dir, args.task)

    if not checkpoints:
        print("No checkpoints found. Nothing to evaluate.")
        return

    print(f"Evaluating {len(checkpoints)} checkpoint(s) on {device}...\n")

    results: list[dict[str, Any]] = []
    for ckpt_path, task in checkpoints:
        model_name = extract_model_name(ckpt_path, task)
        print(f"[{task.upper()}] {model_name} ({ckpt_path.name})")

        if task == "classification":
            result = evaluate_classification(ckpt_path, device, args.batch_size, args.num_workers)
        else:
            result = evaluate_segmentation(ckpt_path, device, args.batch_size, args.num_workers)

        results.append(result)

    # Save all results to disk.
    save_results(results, args.output_dir)
    print(f"\nResults saved to {args.output_dir}/")

    # Print summary table.
    _print_summary(results)


if __name__ == "__main__":
    main()
