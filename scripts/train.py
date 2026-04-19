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

Hydra composes a config from configs/ YAML files. CLI overrides replace any value:
    model=vit_b16          → loads configs/model/vit_b16.yaml
    training.lr=3e-4       → overrides the learning rate
    training.max_epochs=30 → overrides max epochs

Keras equivalent of this whole script:
    model.compile(optimizer=AdamW(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=50,
              callbacks=[checkpoint, early_stop, lr_mon])
    model.evaluate(test_ds)

But here Lightning handles the training loop, device placement, mixed precision,
and logging — while Hydra manages all configuration and MLflow tracks experiments.
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

    # 1. Reproducibility — fixes random seeds for torch, numpy, python random.
    #    Keras equivalent: tf.random.set_seed(42)
    L.seed_everything(cfg.experiment.seed, workers=True)

    # 2. DataModule — loads SDNET2018 splits and creates train/val/test DataLoaders.
    #    Keras equivalent: tf.keras.utils.image_dataset_from_directory(...)
    dm = CrackClassificationDM(
        data_dir=cfg.dataset.root,
        split_file="configs/splits/sdnet2018_split.json",
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.dataset.image_size,
        aug_preset=cfg.aug.name,
    )

    # 3. LightningModule — wraps the timm model + loss + metrics + optimizer.
    #    Keras equivalent: model = Sequential([ResNet50(...), Dense(2)])
    #                      model.compile(optimizer=AdamW, loss=..., metrics=[...])
    model = CrackClassifier(
        model_name=cfg.model.timm_name,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.model.num_classes,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )

    # 4. Loggers — dual logging to MLflow (interactive tracking) and CSV (portable export).
    #    CSVLogger produces metrics.csv + hparams.yaml that can be synced to MLflow
    #    from any environment (e.g. Kaggle) that lacks a tracking server.
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

    # 5. Callbacks — Lightning callbacks ≈ Keras callbacks, same concept.
    callbacks = [
        # ModelCheckpoint ≈ tf.keras.callbacks.ModelCheckpoint
        # Saves the best model (lowest val_loss) to disk.
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename=cfg.training.checkpoint.filename,
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=cfg.training.checkpoint.save_top_k,
        ),
        # EarlyStopping ≈ tf.keras.callbacks.EarlyStopping
        # Stops training when val_loss stops improving for `patience` epochs.
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
        ),
        # LearningRateMonitor — logs LR at each step to the logger (no Keras equivalent).
        LearningRateMonitor(logging_interval="step"),
    ]

    # 6. Trainer — the engine that runs the training/validation/test loops.
    #    Keras equivalent: model.fit(epochs=N) but with built-in support for
    #    mixed precision, multi-GPU, gradient clipping, and more.
    #    precision="16-mixed" ≈ tf.keras.mixed_precision.set_global_policy("mixed_float16")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,
    )

    # 7. Train — Trainer.fit() ≈ model.fit()
    trainer.fit(model, datamodule=dm)

    # 8. Test — evaluate on held-out test set using the best checkpoint.
    #    Keras equivalent: model.load_weights(best_ckpt); model.evaluate(test_ds)
    #    ckpt_path="best" tells Lightning to load the best checkpoint from ModelCheckpoint.
    test_results = trainer.test(model, datamodule=dm, ckpt_path="best")

    # 9. Log the best checkpoint path to MLflow for easy retrieval.
    best_path = trainer.checkpoint_callback.best_model_path
    if best_path and mlflow_logger.experiment:
        mlflow_logger.experiment.log_param(
            mlflow_logger.run_id,
            "best_checkpoint_path",
            best_path,
        )

    # 10. Save portable outputs to CSVLogger version dir.
    #     These files make the run self-contained and downloadable from Kaggle.
    version_dir = Path(csv_logger.log_dir)

    # Resolved Hydra config — full snapshot of all settings for this run
    OmegaConf.save(cfg, version_dir / "config.yaml")

    # Summary — best checkpoint path + final test metrics
    summary = {
        "best_checkpoint_path": best_path,
        "test_metrics": test_results[0] if test_results else {},
    }
    (version_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    train()
