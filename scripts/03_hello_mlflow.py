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

"""MLflow hello world: log parameters, metrics, and artifacts to a local experiment.

Validates that MLflow tracking works locally before integrating with the training
pipeline. Creates an experiment, logs dummy training data, saves a text artifact,
and retrieves everything back via search_runs().

MLflow is the experiment tracker for this project — similar role to TensorBoard in
Keras/TF, but also tracks parameters, artifacts, and model versions.

Keras equivalent:
    TensorBoard logs metrics via `tf.keras.callbacks.TensorBoard(log_dir=...)`.
    MLflow does the same via `mlflow.log_metric()`, plus it tracks hyperparams
    and files (artifacts) that TensorBoard doesn't handle natively.

Usage:
    uv run python scripts/03_hello_mlflow.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
from rich.console import Console
from rich.table import Table

from src.utils.tracking import setup_mlflow

console = Console()


def main() -> None:
    """Run MLflow logging demo."""
    experiment_name = "hello-mlflow-test"
    setup_mlflow(experiment_name=experiment_name)

    # mlflow.start_run() ~ creating a new TensorBoard log directory
    # All log_param / log_metric calls inside the context go to this run.
    with mlflow.start_run(run_name="demo-run") as run:
        run_id = run.info.run_id
        console.print(f"\n[bold green]Started MLflow run:[/] {run_id}\n")

        # --- Log parameters (hyperparameters) ---
        # mlflow.log_param() ~ writing to a config file that TensorBoard doesn't natively support
        params = {
            "model": "resnet50",
            "lr": 1e-4,
            "batch_size": 32,
            "optimizer": "adamw",
            "max_epochs": 50,
        }
        mlflow.log_params(params)
        console.print("[bold]Logged parameters:[/]")
        for k, v in params.items():
            console.print(f"  {k} = {v}")

        # --- Log metrics over steps ---
        # mlflow.log_metric(key, value, step=i) ~ tf.summary.scalar() in TensorBoard
        # The `step` param lets MLflow plot metrics over time, just like TensorBoard epochs.
        console.print("\n[bold]Logging metrics for 5 steps:[/]")
        for step in range(5):
            loss = 1.0 / (step + 1)  # Decreasing dummy loss
            accuracy = 0.5 + step * 0.1  # Increasing dummy accuracy
            mlflow.log_metric("train_loss", loss, step=step)
            mlflow.log_metric("train_accuracy", accuracy, step=step)
            console.print(f"  step {step}: loss={loss:.4f}, accuracy={accuracy:.2f}")

        # --- Log an artifact (file) ---
        # mlflow.log_artifact() saves any file alongside the run — model weights,
        # plots, config snapshots, etc. TensorBoard has no direct equivalent.
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "experiment_notes.txt"
            artifact_path.write_text(
                "Hello MLflow!\n"
                "This artifact was logged from scripts/03_hello_mlflow.py.\n"
                "In a real run, we'd log model checkpoints and plots here.\n"
            )
            mlflow.log_artifact(str(artifact_path))
            console.print(f"\n[bold]Logged artifact:[/] {artifact_path.name}")

    # --- Verify: retrieve the run via search_runs() ---
    # mlflow.search_runs() returns a pandas DataFrame of all runs in an experiment.
    # This is useful for comparing experiments programmatically.
    console.print("\n[bold cyan]Verifying logged data via search_runs()...[/]\n")
    runs_df = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"run_id = '{run_id}'",
    )

    table = Table(title="Retrieved Run Data")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("run_id", runs_df["run_id"].iloc[0])
    table.add_row("status", runs_df["status"].iloc[0])
    table.add_row("param.model", runs_df["params.model"].iloc[0])
    table.add_row("param.lr", runs_df["params.lr"].iloc[0])
    table.add_row("param.batch_size", runs_df["params.batch_size"].iloc[0])
    table.add_row("metric.train_loss (final)", str(runs_df["metrics.train_loss"].iloc[0]))
    table.add_row("metric.train_accuracy (final)", str(runs_df["metrics.train_accuracy"].iloc[0]))

    console.print(table)
    console.print("\n[bold green]MLflow hello world complete![/]")
    console.print("Run [bold]mlflow ui --backend-store-uri mlruns[/] to view in the dashboard.\n")


if __name__ == "__main__":
    main()
