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

"""Segmentation training script — Hydra + Lightning + MLflow.

Usage:
    uv run python scripts/train_seg.py model=unet_resnet50 dataset=crackseg9k aug=medium
    uv run python scripts/train_seg.py model=unet_swin_tiny dataset=crackseg9k aug=heavy
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.data.segmentation_dm import CrackSegmentationDM
from src.models.segmentation_module import CrackSegmentor


@hydra.main(version_base=None, config_path="../configs", config_name="config_seg")
def train(cfg: DictConfig) -> None:
    """Train a segmentation model with the given Hydra config."""

    L.seed_everything(cfg.experiment.seed, workers=True)

    dm = CrackSegmentationDM(
        data_dir=cfg.dataset.root,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.dataset.image_size,
        aug_preset=cfg.aug.name,
        val_ratio=cfg.dataset.split.val,
        test_ratio=cfg.dataset.split.test,
        seed=cfg.experiment.seed,
    )

    model = CrackSegmentor(
        encoder_name=cfg.model.encoder_name,
        decoder=cfg.model.decoder,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    run_name = f"{cfg.model.name}_{cfg.experiment.purpose}"
    if cfg.aug.name != "medium":
        run_name = f"{run_name}_{cfg.aug.name}"

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment.name,
        tracking_uri=cfg.experiment.tracking_uri,
        run_name=run_name,
        log_model=False,
    )
    csv_logger = CSVLogger(
        save_dir=cfg.paths.results,
        name=cfg.experiment.name,
    )
    loggers = [mlflow_logger, csv_logger]

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename=cfg.training.checkpoint.filename,
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=cfg.training.checkpoint.save_top_k,
        ),
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            min_delta=cfg.training.early_stopping.min_delta,
            mode=cfg.training.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path and mlflow_logger.experiment:
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id,
            "best_checkpoint_path",
            best_path,
        )

    version_dir = Path(csv_logger.log_dir)
    OmegaConf.save(cfg, version_dir / "config.yaml")
    summary = {
        "best_checkpoint_path": best_path,
        "test_metrics": test_results[0] if test_results else {},
    }
    (version_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
