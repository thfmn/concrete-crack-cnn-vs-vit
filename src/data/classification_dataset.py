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

"""Dataset for SDNET2018 binary crack classification.

PyTorch equivalent of tf.data.Dataset.map() + tf.keras.utils.image_dataset_from_directory():
- In Keras you call image_dataset_from_directory() which handles loading + labelling.
- In PyTorch you subclass Dataset and implement __getitem__ (called per-sample, like .map())
  and __len__. A DataLoader then wraps the Dataset to add batching, shuffling, and
  multi-process prefetching (like tf.data.batch().prefetch()).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import albumentations as A
    import torch


class CrackClassificationDataset(Dataset):
    """Map-style dataset that loads images and returns (image_tensor, label).

    Each call to __getitem__ loads one image from disk, applies an albumentations
    transform, and returns the result. This is the PyTorch equivalent of
    tf.data.Dataset.map(load_and_preprocess) — the key difference is that PyTorch
    does this lazily per-sample instead of building a graph.

    Args:
        file_paths: Absolute paths to image files.
        labels: Integer labels (0 = uncracked, 1 = cracked), same length as file_paths.
        transform: An albumentations Compose pipeline. Must end with ToTensorV2()
            to convert the numpy array to a torch tensor.
    """

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        transform: A.Compose | None = None,
    ) -> None:
        if len(file_paths) != len(labels):
            msg = f"file_paths ({len(file_paths)}) and labels ({len(labels)}) length mismatch"
            raise ValueError(msg)
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one image, apply transform, return (image_tensor, label).

        This is called by the DataLoader for each sample index — similar to
        the function you'd pass to tf.data.Dataset.map().
        """
        # PIL.Image → numpy array (H, W, C) in uint8 [0, 255]
        image = np.array(Image.open(self.file_paths[index]).convert("RGB"))
        label = self.labels[index]

        if self.transform is not None:
            # albumentations expects numpy (H, W, C) and returns {"image": ...}
            image = self.transform(image=image)["image"]

        return image, label
