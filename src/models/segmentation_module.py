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

"""Lightning module for binary crack segmentation."""

from __future__ import annotations

import lightning as L
import segmentation_models_pytorch as smp
import torch
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex


class CrackSegmentor(L.LightningModule):
    """Binary crack segmentor using smp decoders with timm encoders.

    Wraps an ``smp.Unet`` (or other smp decoder) around any timm encoder
    for single-channel binary mask prediction.  Uses a combined Dice + BCE
    loss and tracks IoU (Jaccard) and Dice (F1) metrics.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        decoder: str = "unet",
        pretrained: bool = True,
        num_classes: int = 1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        decoder_cls = self._resolve_decoder(decoder)
        self.model = decoder_cls(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
        )

        # Combined loss: Dice handles class imbalance, BCE provides stable gradients.
        # Both operate on raw logits (from_logits=True is the DiceLoss default).
        self.dice_loss = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()

        # Metrics — BinaryF1Score is mathematically equivalent to Dice for binary tasks.
        self.val_iou = BinaryJaccardIndex()
        self.val_dice = BinaryF1Score()

        self.test_iou = BinaryJaccardIndex()
        self.test_dice = BinaryF1Score()

    @staticmethod
    def _resolve_decoder(name: str) -> type:
        """Map a decoder name string to the corresponding smp class."""
        decoders: dict[str, type] = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "manet": smp.MAnet,
            "linknet": smp.Linknet,
            "pspnet": smp.PSPNet,
        }
        if name not in decoders:
            raise ValueError(f"Unknown decoder '{name}'. Choose from: {sorted(decoders)}")
        return decoders[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits (B, 1, H, W)."""
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Compute combined Dice + BCE loss on one training batch."""
        images, masks = batch
        logits = self(images)
        loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _eval_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        iou: BinaryJaccardIndex,
        dice: BinaryF1Score,
        prefix: str,
    ) -> torch.Tensor:
        """Shared logic for validation and test steps."""
        images, masks = batch
        logits = self(images)
        loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

        # Squeeze channel dim and convert logits to probabilities for metrics.
        probs = logits.squeeze(1).sigmoid()
        targets = masks.squeeze(1).long()

        iou.update(probs, targets)
        dice.update(probs, targets)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_iou", iou, prog_bar=True)
        self.log(f"{prefix}_dice", dice)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One validation step — compute loss + segmentation metrics."""
        return self._eval_step(batch, self.val_iou, self.val_dice, "val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One test step — same as validation but with test_ prefix metrics."""
        return self._eval_step(batch, self.test_iou, self.test_dice, "test")

    def configure_optimizers(self) -> dict:
        """AdamW optimizer with linear warmup + cosine annealing schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
