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

"""Unit tests for the CrackSegmentor Lightning module.

Tests cover:
- Forward pass output shape
- Loss computation via training_step
- Metric logging via validation_step
- Decoder resolution (valid and invalid names)
- Encoder compatibility (native smp and timm-universal encoders)
- configure_optimizers return structure

All tests use small image sizes (32x32 or 64x64) and pretrained=False
for fast execution. No dataset or disk I/O required.
"""

from __future__ import annotations

import pytest
import torch

from src.models.segmentation_module import CrackSegmentor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_batch(
    batch_size: int = 2,
    image_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a dummy (images, masks) batch for testing.

    Returns:
        images: float32 tensor of shape (B, 3, H, W)
        masks:  float32 tensor of shape (B, 1, H, W) with values in {0, 1}
    """
    images = torch.randn(batch_size, 3, image_size, image_size)
    # smp losses expect float masks with shape (B, 1, H, W)
    masks = torch.randint(0, 2, (batch_size, 1, image_size, image_size)).float()
    return images, masks


# ---------------------------------------------------------------------------
# Forward pass tests
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Verify the model produces correct output shapes."""

    def test_output_shape(self) -> None:
        """Forward pass on (2, 3, 224, 224) input returns (2, 1, 224, 224)."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1, 224, 224)

    def test_output_shape_small(self) -> None:
        """Forward pass works with small 32x32 images."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 1, 32, 32)

    def test_output_is_raw_logits(self) -> None:
        """Output should be raw logits (unbounded), not probabilities."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        # Logits can be negative or > 1 (unlike sigmoid output)
        # With random weights, at least some values should be outside [0, 1]
        assert out.min() < 0.0 or out.max() > 1.0, (
            "Output looks like probabilities, expected raw logits"
        )


# ---------------------------------------------------------------------------
# Loss computation tests
# ---------------------------------------------------------------------------


class TestLossComputation:
    """Verify training_step returns a valid scalar loss."""

    def test_training_step_returns_scalar_loss(self) -> None:
        """training_step returns a scalar tensor with loss > 0."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.train()

        batch = _dummy_batch(batch_size=2, image_size=32)
        loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "Loss should be a scalar (0-d tensor)"
        assert loss.item() > 0, "Combined Dice + BCE loss should be > 0"

    def test_training_step_loss_requires_grad(self) -> None:
        """Loss tensor must have requires_grad=True for backprop."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.train()

        batch = _dummy_batch(batch_size=2, image_size=32)
        loss = model.training_step(batch, batch_idx=0)

        assert loss.requires_grad, "Loss must require grad for backpropagation"


# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


class TestMetricComputation:
    """Verify validation_step logs the expected metrics."""

    def test_validation_step_logs_metrics(self) -> None:
        """validation_step should log val_loss, val_iou, and val_dice."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        batch = _dummy_batch(batch_size=2, image_size=32)

        # Lightning's self.log() stores values in self.trainer, but without
        # a trainer we can call the step and check the metric objects directly.
        with torch.no_grad():
            loss = model.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        # Metrics should have been updated (compute returns a value)
        iou_val = model.val_iou.compute()
        dice_val = model.val_dice.compute()

        assert 0.0 <= iou_val.item() <= 1.0, f"IoU out of range: {iou_val}"
        assert 0.0 <= dice_val.item() <= 1.0, f"Dice out of range: {dice_val}"

    def test_test_step_logs_metrics(self) -> None:
        """test_step should update test_iou and test_dice metrics."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        batch = _dummy_batch(batch_size=2, image_size=32)

        with torch.no_grad():
            loss = model.test_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)

        iou_val = model.test_iou.compute()
        dice_val = model.test_dice.compute()

        assert 0.0 <= iou_val.item() <= 1.0
        assert 0.0 <= dice_val.item() <= 1.0


# ---------------------------------------------------------------------------
# Decoder resolution tests
# ---------------------------------------------------------------------------


class TestDecoderResolution:
    """Verify decoder name lookup works correctly."""

    def test_unet_decoder(self) -> None:
        """decoder='unet' should instantiate without error."""
        model = CrackSegmentor(
            encoder_name="resnet18", decoder="unet", pretrained=False
        )
        assert model is not None

    @pytest.mark.parametrize("decoder", ["unetplusplus", "fpn", "deeplabv3plus"])
    def test_valid_decoders(self, decoder) -> None:
        """All supported decoder names should instantiate without error."""
        model = CrackSegmentor(
            encoder_name="resnet18", decoder=decoder, pretrained=False
        )
        assert model is not None

    def test_invalid_decoder_raises_valueerror(self) -> None:
        """decoder='invalid' should raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown decoder 'invalid'"):
            CrackSegmentor(
                encoder_name="resnet18", decoder="invalid", pretrained=False
            )


# ---------------------------------------------------------------------------
# Encoder compatibility tests
# ---------------------------------------------------------------------------


class TestEncoderCompatibility:
    """Verify both native smp and timm-universal encoders work."""

    def test_native_smp_encoder(self) -> None:
        """resnet50 is a native smp encoder — should instantiate and forward."""
        model = CrackSegmentor(encoder_name="resnet50", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 1, 32, 32)

    def test_timm_universal_encoder(self) -> None:
        """tu-convnext_tiny uses timm-universal prefix — should work via smp."""
        model = CrackSegmentor(
            encoder_name="tu-convnext_tiny", pretrained=False
        )
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 1, 64, 64)

    def test_resnet18_small_encoder(self) -> None:
        """resnet18 — smallest ResNet, useful for fast testing."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 1, 32, 32)


# ---------------------------------------------------------------------------
# configure_optimizers tests
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    """Verify optimizer and scheduler configuration."""

    def test_returns_optimizer_and_scheduler(self) -> None:
        """configure_optimizers returns dict with 'optimizer' and 'lr_scheduler'."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        config = model.configure_optimizers()

        assert isinstance(config, dict)
        assert "optimizer" in config, "Missing 'optimizer' key"
        assert "lr_scheduler" in config, "Missing 'lr_scheduler' key"

    def test_optimizer_is_adamw(self) -> None:
        """Optimizer should be AdamW."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        config = model.configure_optimizers()

        optimizer = config["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_scheduler_has_interval(self) -> None:
        """lr_scheduler dict should contain 'scheduler' and 'interval' keys."""
        model = CrackSegmentor(encoder_name="resnet18", pretrained=False)
        config = model.configure_optimizers()

        lr_config = config["lr_scheduler"]
        assert isinstance(lr_config, dict)
        assert "scheduler" in lr_config
        assert lr_config["interval"] == "epoch"

    def test_lr_matches_hparam(self) -> None:
        """Optimizer base LR should match the lr hyperparameter.

        Note: SequentialLR with LinearLR warmup modifies the *current* LR
        by start_factor, so param_groups[0]["lr"] reflects the warmed-up
        value. The *initial* (base) LR is stored in param_groups[0]["initial_lr"].
        """
        lr = 3e-4
        model = CrackSegmentor(
            encoder_name="resnet18", pretrained=False, lr=lr
        )
        config = model.configure_optimizers()

        optimizer = config["optimizer"]
        # SequentialLR sets initial_lr on each param group
        base_lr = optimizer.param_groups[0]["initial_lr"]
        assert base_lr == pytest.approx(lr)
