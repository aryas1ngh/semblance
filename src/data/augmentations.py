"""Albumentations pipelines for SOP.

Design is driven by the EDA findings (see ``notebooks/deep_eda.ipynb``):

* Wide image size / aspect spread -> random-resized-crop for scale/position
  invariance and a fixed output size.
* The pretrained baseline confused products by colour/texture -> colour jitter
  so the learned embedding relies less on raw colour.
* Real intra-class pose variation -> horizontal flip.

Grayscale images are converted to RGB in the dataset, so transforms always see
3-channel input.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

#: ImageNet channel statistics (the ResNet backbone is pretrained on ImageNet).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224) -> A.Compose:
    """Build the training augmentation pipeline.

    Args:
        img_size: Output square side length in pixels.

    Returns:
        An Albumentations ``Compose`` mapping an ``uint8`` HWC RGB array to a
        normalized ``float32`` CHW tensor.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform(img_size: int = 224) -> A.Compose:
    """Build the deterministic evaluation/inference pipeline.

    Resizes the shortest side to ``img_size * 256/224`` then center-crops, the
    standard ImageNet retrieval preprocessing.

    Args:
        img_size: Output square side length in pixels.

    Returns:
        An Albumentations ``Compose`` mapping an ``uint8`` HWC RGB array to a
        normalized ``float32`` CHW tensor.
    """
    resize = round(img_size * 256 / 224)
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=resize),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
