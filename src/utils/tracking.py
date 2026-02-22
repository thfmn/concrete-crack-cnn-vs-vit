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

"""MLflow experiment tracking utilities.

Sets up local MLflow tracking for the CNN vs ViT benchmark.
All experiments log to a local `mlruns/` directory (self-hosted, no remote server needed).

Keras equivalent:
    In Keras you'd use TensorBoard via `tf.keras.callbacks.TensorBoard(log_dir=...)`.
    MLflow serves a similar purpose but also tracks parameters, artifacts, and model versions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str | Path = "mlruns",
    tags: dict[str, str] | None = None,
) -> str:
    """Configure MLflow tracking for an experiment.

    Args:
        experiment_name: Name of the MLflow experiment (e.g. "crack-classification").
        tracking_uri: Path to the local MLflow tracking directory. Defaults to "mlruns".
        tags: Optional tags to set on the experiment.

    Returns:
        The experiment ID as a string.
    """
    tracking_uri = str(Path(tracking_uri).resolve())
    mlflow.set_tracking_uri(tracking_uri)

    # get_or_create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name, tags=tags)
        logger.info(f"Created MLflow experiment '{experiment_name}' (id={experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (id={experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_hydra_config(cfg: DictConfig) -> None:
    """Log all Hydra config parameters to the active MLflow run.

    Flattens the nested OmegaConf config and logs each key as an MLflow parameter.
    Truncates values to 250 chars (MLflow param value limit).

    Args:
        cfg: The Hydra DictConfig to log.
    """
    flat: dict[str, Any] = {}
    for key, value in _flatten_dict(OmegaConf.to_container(cfg, resolve=True)).items():
        flat[key] = str(value)[:250]

    mlflow.log_params(flat)
    logger.debug(f"Logged {len(flat)} Hydra config params to MLflow")


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dict with dot-separated keys.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for nested keys.
        sep: Separator between nested key levels.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
