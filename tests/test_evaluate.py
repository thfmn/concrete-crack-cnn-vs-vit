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

"""Unit tests for the evaluation script (scripts/evaluate.py).

Tests cover:
- detect_device: auto and explicit device selection
- resolve_family: model name to architecture family mapping
- detect_task: classification vs segmentation checkpoint detection
- extract_model_name: model name extraction from checkpoint hparams
- save_results: JSON and CSV output generation
- discover_checkpoints: checkpoint directory scanning and filtering

All tests use fake checkpoints saved with torch.save() to tmp_path.
No real model weights or datasets are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

# Import the functions under test from the evaluate script.
# sys.path manipulation is not needed because `uv run pytest` includes the
# project root on sys.path automatically.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from evaluate import (  # noqa: E402
    detect_device,
    detect_task,
    discover_checkpoints,
    extract_model_name,
    resolve_family,
    save_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_fake_checkpoint(
    path: Path,
    hparams: dict,
) -> Path:
    """Create a minimal fake Lightning checkpoint file.

    Saves a dict with a ``hyper_parameters`` key, mimicking what
    Lightning writes when calling ``trainer.save_checkpoint()``.

    Args:
        path: File path to write the .ckpt file to.
        hparams: Dictionary of hyperparameters to embed.

    Returns:
        The path that was written to.
    """
    ckpt_data = {"hyper_parameters": hparams}
    torch.save(ckpt_data, path)
    return path


# ---------------------------------------------------------------------------
# detect_device tests
# ---------------------------------------------------------------------------


class TestDetectDevice:
    """Verify device auto-detection and explicit selection."""

    def test_detect_device_auto(self) -> None:
        """detect_device('auto') returns a valid torch.device (cpu or cuda)."""
        device = detect_device("auto")
        assert isinstance(device, torch.device)
        # Should be either cpu or cuda depending on hardware
        assert device.type in ("cpu", "cuda")

    def test_detect_device_cpu(self) -> None:
        """detect_device('cpu') always returns cpu device."""
        device = detect_device("cpu")
        assert isinstance(device, torch.device)
        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# resolve_family tests
# ---------------------------------------------------------------------------


class TestResolveFamily:
    """Verify model name to architecture family mapping."""

    @pytest.mark.parametrize(
        ("model_name", "expected_family"),
        [
            ("resnet50", "cnn"),
            ("tf_efficientnetv2_s", "cnn"),
            ("convnext_tiny", "cnn"),
            ("vit_base_patch16_224", "vit"),
            ("swin_tiny_patch4_window7_224", "vit"),
            ("deit_small_patch16_224", "vit"),
            # Segmentation encoders with timm-universal prefix
            ("tu-tf_efficientnetv2_s", "cnn"),
            ("tu-convnext_tiny", "cnn"),
            ("tu-swin_tiny_patch4_window7_224", "vit"),
        ],
    )
    def test_resolve_family(self, model_name: str, expected_family: str) -> None:
        """Known model names map to the correct architecture family."""
        assert resolve_family(model_name) == expected_family

    def test_resolve_family_unknown(self) -> None:
        """Unknown model names fall back to 'unknown'."""
        assert resolve_family("some_nonexistent_model") == "unknown"


# ---------------------------------------------------------------------------
# detect_task tests
# ---------------------------------------------------------------------------


class TestDetectTask:
    """Verify task detection from checkpoint hyperparameters."""

    def test_detect_task_classification(self, tmp_path: Path) -> None:
        """Checkpoint with 'model_name' in hparams is detected as classification."""
        ckpt_path = _save_fake_checkpoint(
            tmp_path / "cls_model.ckpt",
            hparams={"model_name": "resnet50", "num_classes": 2, "lr": 1e-3},
        )
        assert detect_task(ckpt_path) == "classification"

    def test_detect_task_segmentation(self, tmp_path: Path) -> None:
        """Checkpoint with 'encoder_name' in hparams is detected as segmentation."""
        ckpt_path = _save_fake_checkpoint(
            tmp_path / "seg_model.ckpt",
            hparams={"encoder_name": "resnet50", "decoder": "unet", "lr": 1e-4},
        )
        assert detect_task(ckpt_path) == "segmentation"

    def test_detect_task_raises_on_ambiguous(self, tmp_path: Path) -> None:
        """Checkpoint with neither key raises ValueError."""
        ckpt_path = _save_fake_checkpoint(
            tmp_path / "bad_model.ckpt",
            hparams={"lr": 1e-3, "batch_size": 32},
        )
        with pytest.raises(ValueError, match="Cannot detect task"):
            detect_task(ckpt_path)


# ---------------------------------------------------------------------------
# extract_model_name tests
# ---------------------------------------------------------------------------


class TestExtractModelName:
    """Verify model name extraction from checkpoint hparams."""

    def test_extract_model_name_classification(self, tmp_path: Path) -> None:
        """Classification checkpoint returns the model_name hparam."""
        ckpt_path = _save_fake_checkpoint(
            tmp_path / "cls.ckpt",
            hparams={"model_name": "vit_base_patch16_224", "num_classes": 2},
        )
        name = extract_model_name(ckpt_path, task="classification")
        assert name == "vit_base_patch16_224"

    def test_extract_model_name_segmentation(self, tmp_path: Path) -> None:
        """Segmentation checkpoint returns the encoder_name hparam."""
        ckpt_path = _save_fake_checkpoint(
            tmp_path / "seg.ckpt",
            hparams={"encoder_name": "tu-swin_tiny_patch4_window7_224", "decoder": "unet"},
        )
        name = extract_model_name(ckpt_path, task="segmentation")
        assert name == "tu-swin_tiny_patch4_window7_224"


# ---------------------------------------------------------------------------
# save_results tests
# ---------------------------------------------------------------------------


class TestSaveResults:
    """Verify that save_results writes valid JSON and CSV files."""

    def test_save_results(self, tmp_path: Path) -> None:
        """save_results creates all_results.json, per-model JSON, and CSV files."""
        output_dir = tmp_path / "evaluation"

        # Build mock result dicts matching the structure returned by
        # evaluate_classification and evaluate_segmentation.
        cls_result = {
            "checkpoint": "outputs/checkpoints/resnet50_best.ckpt",
            "task": "classification",
            "model_name": "resnet50",
            "family": "cnn",
            "num_samples": 1000,
            "accuracy": 0.95,
            "f1": 0.94,
            "precision": 0.93,
            "recall": 0.96,
            "confusion_matrix": [[480, 20], [30, 470]],
            "classification_report": {
                "no_crack": {"precision": 0.94, "recall": 0.96, "f1-score": 0.95},
                "crack": {"precision": 0.96, "recall": 0.94, "f1-score": 0.95},
            },
            "inference_time_s": 12.34,
        }
        seg_result = {
            "checkpoint": "outputs/checkpoints/unet_resnet50_best.ckpt",
            "task": "segmentation",
            "model_name": "resnet50",
            "family": "cnn",
            "num_samples": 500,
            "iou": 0.72,
            "dice": 0.84,
            "inference_time_s": 25.67,
        }

        save_results([cls_result, seg_result], output_dir)

        # -- all_results.json --
        all_json_path = output_dir / "all_results.json"
        assert all_json_path.exists(), "all_results.json was not created"
        loaded = json.loads(all_json_path.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 2

        # -- Per-model JSON files --
        models_dir = output_dir / "models"
        assert models_dir.is_dir(), "models/ subdirectory was not created"
        assert (models_dir / "resnet50.json").exists()

        # -- Classification CSV --
        cls_csv_path = output_dir / "comparison_classification.csv"
        assert cls_csv_path.exists(), "comparison_classification.csv was not created"
        cls_df = pd.read_csv(cls_csv_path)
        assert "model" in cls_df.columns
        assert "accuracy" in cls_df.columns
        assert len(cls_df) == 1
        assert cls_df.iloc[0]["accuracy"] == pytest.approx(0.95)

        # -- Segmentation CSV --
        seg_csv_path = output_dir / "comparison_segmentation.csv"
        assert seg_csv_path.exists(), "comparison_segmentation.csv was not created"
        seg_df = pd.read_csv(seg_csv_path)
        assert "iou" in seg_df.columns
        assert "dice" in seg_df.columns
        assert len(seg_df) == 1
        assert seg_df.iloc[0]["iou"] == pytest.approx(0.72)


# ---------------------------------------------------------------------------
# discover_checkpoints tests
# ---------------------------------------------------------------------------


class TestDiscoverCheckpoints:
    """Verify checkpoint directory scanning and task filtering."""

    def test_discover_checkpoints(self, tmp_path: Path) -> None:
        """discover_checkpoints finds .ckpt files and pairs them with tasks."""
        # Create two fake checkpoints: one classification, one segmentation.
        _save_fake_checkpoint(
            tmp_path / "resnet50_best.ckpt",
            hparams={"model_name": "resnet50", "num_classes": 2},
        )
        _save_fake_checkpoint(
            tmp_path / "unet_resnet50_best.ckpt",
            hparams={"encoder_name": "resnet50", "decoder": "unet"},
        )

        # Discover all checkpoints (no filter).
        results = discover_checkpoints(tmp_path, task_filter=None)
        assert len(results) == 2

        # Each result is a (Path, task_string) tuple.
        paths = [r[0] for r in results]
        tasks = [r[1] for r in results]
        assert tmp_path / "resnet50_best.ckpt" in paths
        assert tmp_path / "unet_resnet50_best.ckpt" in paths
        assert "classification" in tasks
        assert "segmentation" in tasks

    def test_discover_checkpoints_with_filter(self, tmp_path: Path) -> None:
        """Task filter returns only checkpoints matching the requested task."""
        _save_fake_checkpoint(
            tmp_path / "cls.ckpt",
            hparams={"model_name": "resnet50"},
        )
        _save_fake_checkpoint(
            tmp_path / "seg.ckpt",
            hparams={"encoder_name": "resnet50"},
        )

        cls_results = discover_checkpoints(tmp_path, task_filter="classification")
        assert len(cls_results) == 1
        assert cls_results[0][1] == "classification"

        seg_results = discover_checkpoints(tmp_path, task_filter="segmentation")
        assert len(seg_results) == 1
        assert seg_results[0][1] == "segmentation"

    def test_discover_checkpoints_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory returns an empty list without raising."""
        results = discover_checkpoints(tmp_path, task_filter=None)
        assert results == []

    def test_discover_checkpoints_skips_unrecognized(self, tmp_path: Path) -> None:
        """Checkpoints with unrecognized hparams are skipped, not errored."""
        _save_fake_checkpoint(
            tmp_path / "good.ckpt",
            hparams={"model_name": "resnet50"},
        )
        _save_fake_checkpoint(
            tmp_path / "bad.ckpt",
            hparams={"lr": 1e-3},  # No model_name or encoder_name
        )

        results = discover_checkpoints(tmp_path, task_filter=None)
        assert len(results) == 1
        assert results[0][0].name == "good.ckpt"
