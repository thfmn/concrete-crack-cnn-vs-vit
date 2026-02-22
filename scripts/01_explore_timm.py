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

"""Explore all 6 target models from timm: parameter counts and output shapes.

Creates each model with num_classes=2 (binary crack/no-crack classification),
runs a forward pass, and prints a comparison table. Also demonstrates how to
inspect and swap the classifier head.

Usage:
    uv run python scripts/01_explore_timm.py
"""

from __future__ import annotations

import timm
import torch
from rich.console import Console
from rich.table import Table

# All 6 benchmark models — 3 CNNs and 3 ViTs.
# Keys are timm model registry names (see timm.list_models()).
MODELS: dict[str, str] = {
    "resnet50": "CNN",
    "tf_efficientnetv2_s": "CNN",
    "convnext_tiny": "CNN",
    "vit_base_patch16_224": "ViT",
    "swin_tiny_patch4_window7_224": "ViT",
    "deit_small_patch16_224": "ViT",
}

NUM_CLASSES: int = 2  # binary: crack vs no-crack


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters.

    Keras equivalent: model.count_params()
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def explore_models() -> list[dict[str, str | int]]:
    """Create each target model and collect metadata.

    Returns:
        List of dicts with model name, family, param count, and output shape.
    """
    # Random input: batch=1, channels=3, height=224, width=224
    # PyTorch uses NCHW layout; Keras/TF defaults to NHWC.
    x = torch.randn(1, 3, 224, 224)

    results: list[dict[str, str | int]] = []

    for model_name, family in MODELS.items():
        # timm.create_model with num_classes=2 replaces the final classification head
        # Keras equivalent: building a model then doing model.layers[-1] = Dense(2)
        # With pretrained=False we skip downloading ImageNet weights (~ weights=None)
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES)
        model.eval()

        with torch.no_grad():
            output = model(x)

        params = count_parameters(model)
        output_shape = tuple(output.shape)

        results.append(
            {
                "name": model_name,
                "family": family,
                "params": params,
                "output_shape": str(output_shape),
            }
        )

    return results


def demonstrate_classifier_api() -> None:
    """Show how to inspect and swap the classifier head via timm's API.

    timm models expose get_classifier() and reset_classifier() — this lets you
    change the number of output classes without rebuilding the entire model.
    In Keras, you'd typically pop the last layer and add a new Dense layer.
    """
    console = Console()
    console.print("\n[bold]Classifier Head API Demo[/bold]")
    console.print("-" * 50)

    model = timm.create_model("resnet50", pretrained=False)

    # get_classifier() returns the final Linear layer
    # Keras equivalent: model.layers[-1] (the Dense output layer)
    original = model.get_classifier()
    console.print(f"  Original classifier: {original}")
    console.print(f"  Output features:     {original.out_features}")

    # reset_classifier(num_classes) swaps in a new Linear head
    # Keras equivalent: replacing the top Dense layer via functional API
    model.reset_classifier(num_classes=NUM_CLASSES)
    new_classifier = model.get_classifier()
    console.print(f"  After reset(2):      {new_classifier}")
    console.print(f"  Output features:     {new_classifier.out_features}")

    # Verify forward pass with new head
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(x)
    console.print(f"  Output shape:        {tuple(output.shape)}")


def print_results_table(results: list[dict[str, str | int]]) -> None:
    """Print a formatted table of model comparisons using rich."""
    console = Console()
    table = Table(title="timm Model Zoo — Target Architectures (num_classes=2)")

    table.add_column("Model", style="cyan")
    table.add_column("Family", style="green")
    table.add_column("Parameters", justify="right", style="magenta")
    table.add_column("Output Shape", style="yellow")

    for row in results:
        table.add_row(
            str(row["name"]),
            str(row["family"]),
            f"{row['params']:,}",
            str(row["output_shape"]),
        )

    console.print()
    console.print(table)


def main() -> None:
    """Explore all 6 target timm models and demonstrate the classifier API."""
    console = Console()
    console.print("[bold]timm Model Zoo Exploration[/bold]")
    console.print(f"timm version: {timm.__version__}")
    console.print(f"torch version: {torch.__version__}")

    # Part 1: Create all 6 models with num_classes=2 and compare
    results = explore_models()
    print_results_table(results)

    # Verify all models produce the expected output shape
    expected_shape = f"(1, {NUM_CLASSES})"
    all_match = all(row["output_shape"] == expected_shape for row in results)
    console.print(f"\nAll models output shape == {expected_shape}: [bold]{all_match}[/bold]")

    # Part 2: Demonstrate get_classifier() and reset_classifier()
    demonstrate_classifier_api()

    console.print(f"\n{'=' * 50}")
    console.print("[bold green]All 6 models explored successfully.[/bold green]")


if __name__ == "__main__":
    main()
