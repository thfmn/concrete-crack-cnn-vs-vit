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

"""Classification training script — Hydra + Lightning + MLflow.

Usage:
    uv run python scripts/train.py model=resnet50 dataset=sdnet2018 aug=medium
    uv run python scripts/train.py model=swin_tiny training.lr=3e-4
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from src.data.classification_dm import CrackClassificationDM
from src.models.classification_module import CrackClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a classification model with the given Hydra config."""

    L.seed_everything(cfg.experiment.seed, workers=True)

    dm = CrackClassificationDM(
        data_dir=cfg.dataset.root,
        split_file="configs/splits/sdnet2018_split.json",
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.dataset.image_size,
        aug_preset=cfg.aug.name,
    )

    model = CrackClassifier(
        model_name=cfg.model.timm_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment.name,
        tracking_uri=cfg.experiment.tracking_uri,
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
