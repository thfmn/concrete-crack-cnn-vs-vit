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

"""Lightning hello world: train a timm model on random data for 2 epochs.

Validates that PyTorch Lightning's training loop works with a timm backbone.
Uses random images + labels (no real dataset needed) to confirm gradient flow
and metric logging.

Lightning is PyTorch's high-level training framework — similar role to Keras'
model.compile() + model.fit() but with explicit control over each step.

Usage:
    uv run python scripts/02_hello_lightning.py
"""

from __future__ import annotations

import lightning as L
import timm
import torch
import torchmetrics
from rich.console import Console
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 2  # binary: crack vs no-crack
NUM_SAMPLES: int = 100  # tiny dataset for smoke-testing
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 16
MAX_EPOCHS: int = 2
LEARNING_RATE: float = 1e-3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DummyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generates random images and labels for testing the training loop.

    torch.utils.data.Dataset is the base class for all datasets in PyTorch.
    Keras equivalent: tf.data.Dataset.from_tensor_slices((x, y))

    You must implement:
        __len__  — how many samples (like len(dataset) in Keras generators)
        __getitem__ — return one (input, label) pair by index
    """

    def __init__(self, num_samples: int, num_classes: int, image_size: int) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Random image: (C, H, W) — PyTorch uses channels-first (NCHW)
        # Keras default is channels-last (NHWC)
        image = torch.randn(3, self.image_size, self.image_size)

        # Random label: integer in [0, num_classes)
        # torch.randint ~ tf.random.uniform(shape=[], maxval=N, dtype=tf.int64)
        label = torch.randint(0, self.num_classes, (1,)).squeeze(0)

        return image, label


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------
class ClassificationModule(L.LightningModule):
    """Wraps a timm model in Lightning's training abstraction.

    Lightning organises training into methods that map to Keras concepts:
        __init__          ~ building the model (tf.keras.Model subclass __init__)
        training_step     ~ what happens inside model.fit() per batch
        validation_step   ~ what happens inside model.evaluate() per batch
        configure_optimizers ~ model.compile(optimizer=...)

    The Trainer then calls these methods automatically — like Keras' model.fit()
    handles the epoch/batch loop, but here you see each piece explicitly.
    """

    def __init__(self, model_name: str, num_classes: int, lr: float) -> None:
        super().__init__()
        # save_hyperparameters() stores constructor args in self.hparams and
        # logs them automatically. Keras has no direct equivalent — you'd
        # manually pass config dicts to model.save().
        self.save_hyperparameters()

        # Create timm model (pretrained=False — no weight download)
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        # Cross-entropy loss: combines LogSoftmax + NLLLoss in one step.
        # Keras equivalent: tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # torchmetrics handles metric accumulation across batches and devices.
        # Keras equivalent: tf.keras.metrics.SparseCategoricalAccuracy()
        # task="binary" because num_classes=2.
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — just delegates to the timm model.

        In Keras, this is model.call() or model.__call__().
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Process one training batch: forward pass, loss, metrics.

        Keras equivalent: this is what happens inside model.fit() for each batch.
        In Keras, the loss/metrics are computed implicitly by model.compile();
        here we do it explicitly.

        Lightning auto-calls loss.backward() and optimizer.step() after this
        returns — no need to write the backward pass yourself.
        """
        images, labels = batch
        logits = self(images)  # calls forward()
        loss = self.loss_fn(logits, labels)

        # For BinaryAccuracy, extract probability of class 1 via softmax
        probs = torch.softmax(logits, dim=1)[:, 1]
        self.train_acc(probs, labels)

        # self.log() records the value for Lightning's progress bar and loggers.
        # Keras equivalent: the metrics dict returned by model.fit().
        # on_step=True logs each batch; on_epoch=True logs the epoch average.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Returning the loss tells Lightning to call loss.backward() automatically.
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Process one validation batch.

        Keras equivalent: what happens inside model.evaluate() per batch.
        Lightning automatically sets model.eval() and torch.no_grad() before
        calling this — you don't need to do it yourself (unlike raw PyTorch).
        """
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        self.val_acc(probs, labels)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define the optimizer.

        Keras equivalent: model.compile(optimizer=tf.keras.optimizers.AdamW(lr=...))
        In Lightning, you return the optimizer object; the Trainer handles the
        step/zero_grad cycle.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Train a ResNet-50 on random data for 2 epochs to verify Lightning works."""
    console = Console()
    console.print("[bold]Lightning Hello World — Training Loop[/bold]")
    console.print(f"torch:     {torch.__version__}")
    console.print(f"lightning: {L.__version__}")

    # --- Data ---
    # In Keras: tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    # In PyTorch: wrap a Dataset in a DataLoader which handles batching,
    # shuffling, and multi-process loading.
    train_ds = DummyDataset(NUM_SAMPLES, NUM_CLASSES, IMAGE_SIZE)
    val_ds = DummyDataset(NUM_SAMPLES // 2, NUM_CLASSES, IMAGE_SIZE)

    # DataLoader ~ tf.data.Dataset.batch().prefetch()
    # num_workers=0 here because the data is in-memory random tensors.
    # For real datasets we'd use num_workers=4, pin_memory=True.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # --- Model ---
    model = ClassificationModule(
        model_name="resnet50",
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
    )

    # --- Trainer ---
    # Trainer(max_epochs=2) ~ model.fit(epochs=2)
    # It handles the train/val loop, progress bar, logging, and device placement.
    #
    # Callbacks in Lightning are like Keras callbacks:
    #   - ModelCheckpoint  ~ tf.keras.callbacks.ModelCheckpoint
    #   - EarlyStopping    ~ tf.keras.callbacks.EarlyStopping
    #   - LearningRateMonitor ~ tf.keras.callbacks.ReduceLROnPlateau (logging part)
    # We skip callbacks here to keep it minimal.
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",  # force CPU for this smoke test
        enable_checkpointing=False,  # no checkpoint files for demo
        logger=False,  # disable file logging for demo
        enable_progress_bar=True,
    )

    # trainer.fit() runs the full training + validation loop.
    # Keras equivalent: model.fit(x_train, y_train, validation_data=(x_val, y_val))
    console.print("\n[bold]Starting training...[/bold]")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --- Results ---
    console.print(f"\n{'=' * 60}")
    console.print("[bold green]Training complete![/bold green]")

    # Access logged metrics from the trainer's callback_metrics dict.
    metrics = trainer.callback_metrics
    console.print(f"  Final train_loss_epoch: {metrics.get('train_loss_epoch', 'N/A')}")
    console.print(f"  Final train_acc:        {metrics.get('train_acc', 'N/A')}")
    console.print(f"  Final val_loss:         {metrics.get('val_loss', 'N/A')}")
    console.print(f"  Final val_acc:          {metrics.get('val_acc', 'N/A')}")


if __name__ == "__main__":
    main()
