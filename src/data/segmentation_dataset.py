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

"""Dataset for CrackSeg9k semantic segmentation.

PyTorch equivalent of a tf.data pipeline with paired image+mask loading:
- In Keras/TF you'd build a tf.data.Dataset that loads (image, mask) pairs and
  applies tf.image transforms. Mask transforms must match spatial transforms
  (flip, rotate) but skip pixel transforms (brightness, contrast).
- In PyTorch you subclass Dataset and implement __getitem__. Albumentations
  handles synchronized image+mask transforms automatically — pass both to
  transform(image=img, mask=msk) and spatial transforms are applied identically
  to both, while pixel-level transforms only affect the image.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import albumentations as A


class CrackSegmentationDataset(Dataset):
    """Map-style dataset that loads image-mask pairs for segmentation.

    Each call to __getitem__ loads one image and its corresponding mask from
    disk, applies an albumentations transform to both simultaneously, and
    returns (image_tensor, mask_tensor). This is the PyTorch equivalent of
    tf.data.Dataset.map(load_and_preprocess) — but lazy, per-sample.

    Mask preprocessing:
        CrackSeg9k masks have anti-aliased edges (values 0–255). We threshold
        at 127 to produce clean binary masks (0=background, 1=crack) with
        dtype torch.long, as required by PyTorch's CrossEntropyLoss.

    Args:
        image_paths: Absolute paths to input images.
        mask_paths: Absolute paths to corresponding grayscale masks.
            Must be the same length and order as image_paths.
        transform: An albumentations Compose pipeline. Spatial transforms
            (flip, rotate, resize) are applied to both image and mask.
            Pixel transforms (brightness, contrast) only affect the image.
    """

    def __init__(
        self,
        image_paths: list[Path],
        mask_paths: list[Path],
        transform: A.Compose | None = None,
    ) -> None:
        if len(image_paths) != len(mask_paths):
            msg = (
                f"image_paths ({len(image_paths)}) and "
                f"mask_paths ({len(mask_paths)}) length mismatch"
            )
            raise ValueError(msg)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one image+mask pair, apply transform, return tensors.

        This is called by the DataLoader for each sample index — similar to
        the function you'd pass to tf.data.Dataset.map().

        Returns:
            (image, mask) where image is float32 [C, H, W] and mask is
            int64 [H, W] with values in {0, 1}.
        """
        # Load image as RGB numpy array (H, W, C) uint8
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))

        # Load mask as grayscale numpy array (H, W) uint8 [0–255]
        mask = np.array(Image.open(self.mask_paths[index]).convert("L"))

        # Binarize: threshold at 127 → {0, 1} (handles anti-aliased edges)
        mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            # albumentations applies spatial transforms (flip, rotate, resize) to both
            # image and mask identically; pixel transforms (brightness, contrast) only
            # affect the image. Just pass mask= and it works.
            result = self.transform(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        # Convert mask to torch.long for CrossEntropyLoss / BCEWithLogitsLoss
        # (image is already a float32 tensor from ToTensorV2)
        mask = torch.as_tensor(mask, dtype=torch.long)

        return image, mask
