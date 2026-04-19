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

"""Lightweight project dashboard server.

Serves a single-page cockpit UI and exposes JSON API endpoints
for ticket tracking, MLflow metrics, git activity, and pipeline status.

Usage:
    uv run python scripts/dashboard.py
    uv run python scripts/dashboard.py --port 9000
"""

from __future__ import annotations

import argparse
import http.server
import json
import re
import subprocess
import webbrowser
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import yaml

# Resolve project root (parent of scripts/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"
TICKETS_PATH: Path = DASHBOARD_DIR / "tickets.yaml"
MLFLOW_DB: Path = PROJECT_ROOT / "mlflow.db"
MLFLOW_URI: str = f"sqlite:///{MLFLOW_DB}"

# Components to check for pipeline status
COMPONENTS: dict[str, str] = {
    "split": "src/data/split.py",
    "augmentation": "src/data/augmentation.py",
    "classification_dataset": "src/data/classification_dataset.py",
    "classification_dm": "src/data/classification_dm.py",
    "segmentation_dataset": "src/data/segmentation_dataset.py",
    "segmentation_dm": "src/data/segmentation_dm.py",
    "classification_module": "src/models/classification_module.py",
    "segmentation_module": "src/models/segmentation_module.py",
    "train_cls": "scripts/train.py",
    "train_seg": "scripts/train_seg.py",
    "evaluate": "scripts/evaluate.py",
    "export_onnx": "scripts/export_onnx.py",
}

# Ticket-reference regex (e.g., CLS-2, DATA-11)
TICKET_REF_RE: re.Pattern[str] = re.compile(r"[A-Z]+-\d+")


def get_tickets() -> dict[str, Any]:
    """Read tickets.yaml and compute per-epic progress."""
    if not TICKETS_PATH.exists():
        return {"epics": [], "summary": {"done": 0, "in_progress": 0, "todo": 0, "total": 0}}

    with open(TICKETS_PATH) as f:
        data = yaml.safe_load(f)

    total_done = 0
    total_in_progress = 0
    total_todo = 0

    for epic in data.get("epics", []):
        counts = {"done": 0, "in_progress": 0, "todo": 0}
        for ticket in epic.get("tickets", []):
            status = ticket.get("status", "todo")
            counts[status] = counts.get(status, 0) + 1
        epic["progress"] = {
            **counts,
            "total": sum(counts.values()),
        }
        total_done += counts["done"]
        total_in_progress += counts["in_progress"]
        total_todo += counts["todo"]

    data["summary"] = {
        "done": total_done,
        "in_progress": total_in_progress,
        "todo": total_todo,
        "total": total_done + total_in_progress + total_todo,
    }
    return data


def get_mlflow_data() -> dict[str, Any]:
    """Query MLflow experiments and runs from SQLite database."""
    if not MLFLOW_DB.exists():
        return {"experiments": [], "has_training_runs": False}

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_URI)
        experiments_data: list[dict[str, Any]] = []
        has_training_runs = False

        for exp in client.search_experiments():
            # Skip the default experiment if it has no runs
            runs_data: list[dict[str, Any]] = []
            for run in client.search_runs(experiment_ids=[exp.experiment_id]):
                start_ms = run.info.start_time or 0
                end_ms = run.info.end_time or 0
                duration_s = (end_ms - start_ms) / 1000.0 if end_ms > start_ms else 0

                # Check if this looks like a real training run (has val metrics)
                metrics = dict(run.data.metrics)
                if any(k.startswith("val_") for k in metrics):
                    has_training_runs = True

                runs_data.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name or run.info.run_id[:8],
                    "status": run.info.status,
                    "start_time": (
                        datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
                        if start_ms
                        else None
                    ),
                    "duration_s": round(duration_s, 1),
                    "params": dict(run.data.params),
                    "metrics": {k: round(v, 4) for k, v in metrics.items()},
                })

            experiments_data.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "runs": runs_data,
            })

        return {"experiments": experiments_data, "has_training_runs": has_training_runs}

    except Exception as e:
        return {"experiments": [], "has_training_runs": False, "error": str(e)}


def get_mlflow_history() -> dict[str, Any]:
    """Return per-epoch metric history for the most recent (or running) training run."""
    if not MLFLOW_DB.exists():
        return {"run": None}

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW_URI)

        # Find the most recent run that has val_ metrics (real training run).
        # Prefer RUNNING runs over finished ones.
        best_run = None
        for exp in client.search_experiments():
            for run in client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10,
            ):
                has_val = any(k.startswith("val_") for k in run.data.metrics)
                if not has_val:
                    continue
                # Prefer a RUNNING run; otherwise take the most recent finished one
                if run.info.status == "RUNNING":
                    best_run = run
                    break
                if best_run is None:
                    best_run = run
            if best_run and best_run.info.status == "RUNNING":
                break

        if best_run is None:
            return {"run": None}

        # Fetch metric history for key training/validation metrics
        metric_keys = [
            "train_loss", "val_loss", "val_acc", "val_f1", "val_precision", "val_recall",
        ]
        histories: dict[str, list[dict[str, float]]] = {}
        for key in metric_keys:
            try:
                history = client.get_metric_history(best_run.info.run_id, key)
                histories[key] = [
                    {"step": m.step, "value": round(m.value, 4)} for m in history
                ]
            except Exception:
                continue

        return {
            "run": {
                "run_id": best_run.info.run_id,
                "run_name": best_run.info.run_name or best_run.info.run_id[:8],
                "status": best_run.info.status,
                "model": best_run.data.params.get("model_name", "unknown"),
                "metrics": histories,
            }
        }

    except Exception as e:
        return {"run": None, "error": str(e)}


def get_git_data() -> dict[str, Any]:
    """Get recent git log and branch info."""
    try:
        # Get current branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        branch = branch_result.stdout.strip()

        # Get recent commits
        log_result = subprocess.run(
            ["git", "log", "--format=%H|%h|%s|%an|%ar|%aI", "-20"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        commits: list[dict[str, str | list[str]]] = []
        for line in log_result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 5)
            if len(parts) < 6:
                continue
            full_hash, short_hash, message, author, relative_time, iso_time = parts

            # Extract ticket references from commit message
            ticket_refs = TICKET_REF_RE.findall(message)

            commits.append({
                "hash": full_hash,
                "short_hash": short_hash,
                "message": message,
                "author": author,
                "relative_time": relative_time,
                "iso_time": iso_time,
                "ticket_refs": ticket_refs,
            })

        return {
            "branch": branch,
            "commits": commits,
            "total_commits": len(commits),
        }

    except Exception as e:
        return {"branch": "unknown", "commits": [], "total_commits": 0, "error": str(e)}


def get_component_status() -> dict[str, Any]:
    """Check filesystem for pipeline component existence."""
    components: dict[str, dict[str, str]] = {}
    for name, rel_path in COMPONENTS.items():
        full_path = PROJECT_ROOT / rel_path
        # A file is "done" if it exists and has meaningful content (> copyright header)
        if full_path.exists() and full_path.stat().st_size > 500:
            status = "done"
        elif full_path.exists():
            status = "partial"
        else:
            status = "missing"
        components[name] = {"status": status, "file": rel_path}

    # Compute pipeline readiness
    cls_components = [
        "classification_dataset",
        "classification_dm",
        "classification_module",
        "train_cls",
        "augmentation",
    ]
    seg_components = [
        "segmentation_dataset",
        "segmentation_dm",
        "segmentation_module",
        "train_seg",
        "augmentation",
    ]

    def pipeline_progress(component_names: list[str]) -> dict[str, int]:
        ready = sum(1 for c in component_names if components[c]["status"] == "done")
        return {"ready": ready, "total": len(component_names), "percent": int(ready / len(component_names) * 100)}

    return {
        "components": components,
        "pipelines": {
            "classification": pipeline_progress(cls_components),
            "segmentation": pipeline_progress(seg_components),
        },
    }


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Routes /api/* to data-gathering functions, everything else to static files."""

    def do_GET(self) -> None:
        api_routes: dict[str, Any] = {
            "/api/tickets": get_tickets,
            "/api/mlflow": get_mlflow_data,
            "/api/mlflow/history": get_mlflow_history,
            "/api/git": get_git_data,
            "/api/status": get_component_status,
        }

        if self.path in api_routes:
            self._json_response(api_routes[self.path]())
        else:
            super().do_GET()

    def _json_response(self, data: dict[str, Any]) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress noisy per-request logging; only log errors."""
        if args and isinstance(args[0], str) and args[0].startswith("GET /api/"):
            return  # Suppress polling noise
        super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project dashboard server")
    parser.add_argument("--port", type=int, default=8050, help="Port to serve on (default: 8050)")
    parser.add_argument(
        "--no-open", action="store_true", help="Don't auto-open browser on start"
    )
    args = parser.parse_args()

    if not DASHBOARD_DIR.exists():
        print(f"Error: dashboard directory not found at {DASHBOARD_DIR}")
        raise SystemExit(1)

    # Serve static files from dashboard/ directory
    handler = partial(DashboardHandler, directory=str(DASHBOARD_DIR))

    server = http.server.HTTPServer(("0.0.0.0", args.port), handler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}")
    print("Press Ctrl+C to stop\n")

    if not args.no_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server")
        server.shutdown()


if __name__ == "__main__":
    main()
