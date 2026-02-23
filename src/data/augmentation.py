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

"""Albumentations augmentation presets for classification and segmentation.

albumentations replaces tf.keras.preprocessing.image.ImageDataGenerator:
- More augmentation types (elastic, grid distortion, CLAHE, coarse dropout, ...)
- Faster (numpy/OpenCV backend instead of TF ops)
- Supports simultaneous image+mask transforms for segmentation — spatial transforms
  (flip, rotate, distort) are applied identically to both image and mask.

Three training presets with increasing regularization strength:
- **light**: minimal augmentation — use when dataset is large or overfitting is low
- **medium**: moderate augmentation — good default for most experiments
- **heavy**: aggressive augmentation — use for small datasets or heavy overfitting
"""

from __future__ import annotations

from typing import Literal

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization — standard for pretrained timm models.
# After normalization, pixel values are roughly zero-mean, unit-variance (~[-2.12, 2.64]).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

Preset = Literal["light", "medium", "heavy"]


def get_train_transforms(preset: Preset, image_size: int) -> A.Compose:
    """Build a training augmentation pipeline for the given preset.

    All presets resize to image_size×image_size, normalize with ImageNet stats,
    and convert to a PyTorch tensor. Higher presets add progressively more
    augmentation for stronger regularization.

    The returned Compose works for both classification and segmentation:
    - Classification: ``transform(image=img)["image"]``
    - Segmentation: ``result = transform(image=img, mask=msk)`` — spatial transforms
      (flip, rotate, elastic, grid distortion) are applied identically to both.

    Args:
        preset: Augmentation strength — "light", "medium", or "heavy".
        image_size: Target spatial size (e.g. 224 for timm models).

    Returns:
        An albumentations Compose pipeline ending with Normalize + ToTensorV2.
    """
    # --- Spatial transforms (applied to both image and mask) ---
    spatial: list[A.BasicTransform] = [A.Resize(image_size, image_size)]

    if preset in ("light", "medium", "heavy"):
        spatial.append(A.HorizontalFlip(p=0.5))

    if preset in ("medium", "heavy"):
        spatial.append(
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            )
        )

    if preset == "heavy":
        spatial.append(A.ElasticTransform(alpha=1, sigma=50, p=0.3))
        spatial.append(A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3))

    # --- Pixel-level transforms (image only, masks are unchanged) ---
    pixel: list[A.BasicTransform] = []

    if preset in ("medium", "heavy"):
        pixel.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        pixel.append(A.CLAHE(clip_limit=4.0, p=0.3))

    if preset == "heavy":
        pixel.append(
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0,
                p=0.3,
            )
        )

    # --- Finalize: normalize with ImageNet stats → convert to tensor ---
    finalize: list[A.BasicTransform] = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(spatial + pixel + finalize)


def get_val_transforms(image_size: int) -> A.Compose:
    """Build a deterministic validation/test pipeline (no random augmentation).

    Equivalent to Keras ``ImageDataGenerator(rescale=1/255.)`` + model-specific
    preprocessing, but here we resize and normalize with ImageNet stats.

    Args:
        image_size: Target spatial size (e.g. 224 for timm models).

    Returns:
        An albumentations Compose pipeline: Resize + Normalize + ToTensorV2.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
