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

"""Lightning module for binary crack classification.

A LightningModule is the PyTorch equivalent of a compiled Keras model:
- training_step   ~ what happens inside model.fit() per batch
- validation_step ~ what happens during model.evaluate() per batch
- configure_optimizers ~ model.compile(optimizer=...)
- Trainer(max_epochs=N) ~ model.fit(epochs=N)

This module wraps any timm model for binary classification and handles
training, validation, test loops, metrics, and optimizer configuration.
"""

from __future__ import annotations

import lightning as L
import timm
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class CrackClassifier(L.LightningModule):
    """Binary crack classifier wrapping any timm backbone.

    Keras equivalent:
        base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
        model = tf.keras.Sequential([base, GlobalAveragePooling2D(), Dense(2)])
        model.compile(optimizer=AdamW(lr), loss="sparse_categorical_crossentropy")

    But here timm handles the backbone + classifier head in one call, and
    Lightning manages the training loop, logging, and device placement.

    Args:
        model_name: timm model key (e.g. "resnet50", "vit_base_patch16_224").
        pretrained: Load ImageNet pretrained weights (like Keras weights="imagenet").
        num_classes: Number of output classes (2 for binary crack/no-crack).
        lr: Learning rate for AdamW optimizer.
        weight_decay: L2 regularization strength for AdamW.
        warmup_epochs: Number of epochs for linear warmup before cosine decay.
        max_epochs: Total training epochs (needed for cosine schedule calculation).
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
    ) -> None:
        super().__init__()
        # save_hyperparameters stores all __init__ args in self.hparams
        # and logs them automatically when using a Lightning logger.
        # Keras equivalent: these would be passed to model.compile() and logged manually.
        self.save_hyperparameters()

        # timm.create_model builds a complete model (backbone + classifier head).
        # Keras equivalent: tf.keras.applications.ResNet50(weights="imagenet", classes=2)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        # torchmetrics handle metric accumulation across batches automatically.
        # Keras equivalent: metrics=["accuracy"] in model.compile(), but here we
        # define them explicitly and get more control over aggregation.
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits (B, num_classes).

        In PyTorch, models return raw logits (unnormalized scores).
        The loss function (cross_entropy) applies softmax internally.
        Keras equivalent: model(x) or model.predict(x), but Keras returns
        probabilities by default if using softmax activation.
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One training step — compute loss and log it.

        This is what happens inside model.fit() for each batch in Keras.
        Lightning calls this automatically for every batch in the train DataLoader.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        # self.log() ~ Keras verbose=1 output, but also writes to TensorBoard/MLflow
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _eval_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        acc: BinaryAccuracy,
        f1: BinaryF1Score,
        precision: BinaryPrecision,
        recall: BinaryRecall,
        prefix: str,
    ) -> torch.Tensor:
        """Shared logic for validation and test steps."""
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Convert logits to probabilities for the positive class (class 1).
        # For binary metrics, torchmetrics expects probabilities or predictions,
        # not raw logits. We take softmax then select class-1 probability.
        probs = logits.softmax(dim=1)[:, 1]

        acc.update(probs, labels)
        f1.update(probs, labels)
        precision.update(probs, labels)
        recall.update(probs, labels)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
        self.log(f"{prefix}_f1", f1)
        self.log(f"{prefix}_precision", precision)
        self.log(f"{prefix}_recall", recall)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """One validation step — compute loss + metrics.

        Lightning automatically sets model.eval() and torch.no_grad() before
        calling this. Keras equivalent: the per-batch logic in model.evaluate().
        """
        return self._eval_step(
            batch, self.val_acc, self.val_f1, self.val_precision, self.val_recall, "val"
        )

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One test step — same as validation but with test_ prefix metrics."""
        return self._eval_step(
            batch, self.test_acc, self.test_f1, self.test_precision, self.test_recall, "test"
        )

    def configure_optimizers(self) -> dict:
        """Set up AdamW optimizer with cosine LR schedule + linear warmup.

        Keras equivalent:
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
            scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_schedule_fn)
            model.compile(optimizer=optimizer)
            model.fit(..., callbacks=[scheduler])

        In PyTorch Lightning, optimizer + scheduler are returned together from
        this method, and the Trainer handles calling scheduler.step() each epoch.

        torch.optim.AdamW ≈ tf.keras.optimizers.AdamW
        CosineAnnealingLR ≈ tf.keras.callbacks.LearningRateScheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # LinearLR: linearly ramps from start_factor * lr to lr over warmup_epochs.
        # CosineAnnealingLR: decays lr following a cosine curve after warmup.
        # SequentialLR chains them: warmup for N epochs, then cosine for the rest.
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
