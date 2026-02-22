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

"""PyTorch + timm hello world: forward pass through ResNet-50 and ViT-B/16.

Validates that torch and timm are installed correctly by running inference
on random data with one CNN and one ViT model.

Usage:
    uv run python scripts/00_hello_pytorch.py
"""

from __future__ import annotations

import timm
import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model.

    Keras equivalent: model.count_params()
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward_pass(model_name: str) -> None:
    """Load a timm model and run a single forward pass on random data.

    Args:
        model_name: A timm model key (e.g. "resnet50", "vit_base_patch16_224").
    """
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    # timm.create_model() ~ tf.keras.applications.ResNet50()
    # pretrained=False skips downloading ImageNet weights (~ weights=None in Keras)
    model = timm.create_model(model_name, pretrained=False)

    # model.eval() switches to inference mode: disables dropout and freezes
    # batch-norm running stats.  Keras doesn't need this — its layers check
    # the `training` flag passed by model(x, training=False) automatically.
    model.eval()

    # torch.randn creates a tensor of random values from N(0,1)
    # Shape: (batch=1, channels=3, height=224, width=224)
    # NOTE: PyTorch uses NCHW layout; Keras/TF defaults to NHWC.
    x = torch.randn(1, 3, 224, 224)

    # torch.no_grad() disables gradient computation, saving memory during
    # inference.  Keras handles this implicitly — gradients are only tracked
    # inside GradientTape or during model.fit().
    with torch.no_grad():
        output = model(x)

    num_params = count_parameters(model)

    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Parameters:   {num_params:,}")


def main() -> None:
    """Run forward passes for one CNN and one ViT model."""
    print("PyTorch + timm Hello World")
    print(f"torch version: {torch.__version__}")
    print(f"timm version:  {timm.__version__}")

    # CNN representative
    run_forward_pass("resnet50")

    # ViT representative
    run_forward_pass("vit_base_patch16_224")

    print(f"\n{'=' * 60}")
    print("All forward passes completed successfully.")


if __name__ == "__main__":
    main()
