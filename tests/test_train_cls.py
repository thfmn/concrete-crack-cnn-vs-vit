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

"""Smoke tests for the classification training pipeline.

These tests verify that the full training loop (fit + test + checkpoint + logging)
works end-to-end on CPU with synthetic data and a tiny model.

This mirrors what kaggle/train_cls.py does, just with:
- resnet18 instead of resnet50 (smaller, faster)
- 32x32 images instead of 224x224
- 2 epochs instead of 50
- pretrained=False (no downloads)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _create_synthetic_dataset(root: Path) -> tuple[Path, Path]:
    """Create a minimal SDNET2018-like directory with synthetic images.

    Returns (data_dir, split_file) for use with CrackClassificationDM.
    """
    all_rel_paths: list[str] = []

    # Create 4 images per class across 2 subsets (16 total)
    for subset, cracked, uncracked in [("D", "CD", "UD"), ("P", "CP", "UP")]:
        for cls_dir in [cracked, uncracked]:
            d = root / subset / cls_dir
            d.mkdir(parents=True)
            for i in range(4):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                rel = f"{subset}/{cls_dir}/img_{i}.jpg"
                img.save(root / rel)
                all_rel_paths.append(rel)

    # Split: 8 train, 4 val, 4 test
    split = {
        "train": all_rel_paths[:8],
        "val": all_rel_paths[8:12],
        "test": all_rel_paths[12:],
    }
    split_file = root / "split.json"
    split_file.write_text(json.dumps(split))

    return root, split_file


class TestClassificationTrainingPipeline:
    """End-to-end smoke tests for the classification training pipeline."""

    def test_fit_test_checkpoint_and_csv_metrics(self, tmp_path: Path) -> None:
        """Full training run: fit → test → checkpoint saved → CSV metrics created.

        This is the core smoke test that mirrors kaggle/train_cls.py.
        Uses resnet18 (smallest ResNet) with 32x32 synthetic images for speed.
        """
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger

        from src.data.classification_dm import CrackClassificationDM
        from src.models.classification_module import CrackClassifier

        data_dir, split_file = _create_synthetic_dataset(tmp_path / "data")

        dm = CrackClassificationDM(
            data_dir=data_dir,
            split_file=split_file,
            batch_size=4,
            num_workers=0,
            image_size=32,
            aug_preset="light",
        )

        model = CrackClassifier(
            model_name="resnet18",
            pretrained=False,
            num_classes=2,
            lr=1e-3,
            weight_decay=0.01,
            warmup_epochs=1,
            max_epochs=2,
        )

        ckpt_dir = tmp_path / "checkpoints"
        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        log_dir = tmp_path / "logs"
        logger = CSVLogger(save_dir=str(log_dir), name="test_run")

        trainer = L.Trainer(
            max_epochs=2,
            precision="32",  # CPU doesn't support 16-mixed
            callbacks=[checkpoint_cb],
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
        )

        # Fit — equivalent to Keras model.fit()
        trainer.fit(model, datamodule=dm)

        # Test — equivalent to Keras model.evaluate()
        results = trainer.test(model, datamodule=dm, ckpt_path="best")

        # Verify checkpoint was saved
        assert checkpoint_cb.best_model_path != ""
        assert Path(checkpoint_cb.best_model_path).exists()

        # Verify test results contain expected metrics
        assert len(results) == 1
        assert "test_loss" in results[0]
        assert "test_acc" in results[0]
        assert "test_f1" in results[0]
        assert "test_precision" in results[0]
        assert "test_recall" in results[0]

        # Verify CSVLogger created metrics file
        csv_files = list(log_dir.rglob("metrics.csv"))
        assert len(csv_files) >= 1, f"No metrics.csv found in {log_dir}"

    def test_all_model_configs_valid(self) -> None:
        """Verify the MODELS lookup table in the Kaggle script has valid entries."""
        import timm

        # Same dict as in kaggle/train_cls.py
        models = {
            "resnet50": "resnet50",
            "efficientnetv2_s": "tf_efficientnetv2_s",
            "convnext_tiny": "convnext_tiny",
            "vit_b16": "vit_base_patch16_224",
            "swin_tiny": "swin_tiny_patch4_window7_224",
            "deit_small": "deit_small_patch16_224",
        }

        available = timm.list_models()
        for name, timm_key in models.items():
            assert timm_key in available, (
                f"{name}: timm key '{timm_key}' not found in timm.list_models()"
            )

    def test_kaggle_script_syntax(self) -> None:
        """Verify kaggle/train_cls.py has valid Python syntax."""
        import ast

        script = Path(__file__).parent.parent / "kaggle" / "train_cls.py"
        if not script.exists():
            pytest.skip("kaggle/train_cls.py not found")

        # ast.parse will raise SyntaxError if invalid
        ast.parse(script.read_text())
