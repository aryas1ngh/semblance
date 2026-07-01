"""PyTorch dataset for Stanford Online Products."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class SOPDataset(Dataset[tuple[Tensor, int]]):
    """Serve ``(image, label)`` pairs for an SOP split.

    Args:
        root: Path to the extracted ``Stanford_Online_Products`` directory.
        df: A split DataFrame that already has a contiguous ``label`` column
            (see :func:`src.data.splits.add_contiguous_labels`).
        transform: An Albumentations transform called as ``transform(image=arr)``
            on an ``uint8`` HWC RGB array, returning a CHW tensor under ``"image"``.
        return_index: If ``True``, ``__getitem__`` also returns the row index, so
            embeddings can be mapped back to gallery rows during retrieval.
    """

    def __init__(
        self,
        root: Path,
        df: pd.DataFrame,
        transform,  # noqa: ANN001 -- Albumentations Compose has no precise public type
        *,
        return_index: bool = False,
    ) -> None:
        if "label" not in df.columns:
            raise ValueError("df must have a 'label' column; call add_contiguous_labels first")
        self.root = Path(root)
        self.paths: list[str] = df["path"].tolist()
        self.labels: list[int] = df["label"].astype(int).tolist()
        self.transform = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int] | tuple[Tensor, int, int]:
        with Image.open(self.root / self.paths[idx]) as im:
            arr = np.asarray(im.convert("RGB"))  # force RGB: SOP has a few grayscale images
        image: Tensor = self.transform(image=arr)["image"]
        label = self.labels[idx]
        if self.return_index:
            return image, label, idx
        return image, label
