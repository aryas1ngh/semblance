"""Smoke test for the SOP data pipeline.

Loads one PK batch, verifies its shape and PK composition, and saves an
augmented sample grid. Run:

    python -m src.data.smoke --p 8 --k 4

This is the Phase-1 "done when" check; PK-composition assertions stand in for a
formal test suite until the scaffolding phase.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD, build_train_transform
from src.data.dataset import SOPDataset
from src.data.sampler import PKSampler
from src.data.splits import add_contiguous_labels, load_split

# cv2 spawns its own threads; disable to avoid oversubscription with DataLoader workers.
cv2.setNumThreads(0)


def denormalize(img: Tensor) -> np.ndarray:
    """Undo ImageNet normalization and return an HWC uint8 array for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    arr = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def save_aug_grid(images: Tensor, labels: Tensor, out: Path, p_show: int, k: int) -> None:
    """Save a grid with one class per row and its K augmented samples per column."""
    classes = list(dict.fromkeys(labels.tolist()))[:p_show]  # preserve order, unique
    fig, axes = plt.subplots(len(classes), k, figsize=(k * 2, len(classes) * 2))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(classes):
        member_idx = (labels == cls).nonzero(as_tuple=True)[0][:k]
        for c in range(k):
            ax = axes[r, c]
            if c < len(member_idx):
                ax.imshow(denormalize(images[member_idx[c]]))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"class {cls}", fontsize=9)
    fig.suptitle("SOP PK batch — one class per row, augmented samples across columns", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/Stanford_Online_Products"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--p", type=int, default=8, help="classes per batch")
    parser.add_argument("--k", type=int, default=4, help="samples per class")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/data"))
    args = parser.parse_args()

    df, num_classes = add_contiguous_labels(load_split(args.root, args.split))
    print(f"split={args.split}: {len(df):,} images, {num_classes:,} classes")

    dataset = SOPDataset(args.root, df, build_train_transform(args.img_size))
    sampler = PKSampler(df["label"].to_numpy(), p=args.p, k=args.k, seed=args.seed)
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        dataset, batch_sampler=sampler, num_workers=args.workers
    )
    print(f"sampler: {len(sampler)} batches/epoch of {args.p}x{args.k}={args.p * args.k}")

    images, labels = next(iter(loader))
    expected = (args.p * args.k, 3, args.img_size, args.img_size)
    print(f"batch tensor: shape={tuple(images.shape)} dtype={images.dtype}")

    # --- correctness assertions (PK composition) ---
    counts = Counter(labels.tolist())
    assert tuple(images.shape) == expected, f"expected {expected}, got {tuple(images.shape)}"
    assert len(counts) == args.p, f"expected {args.p} classes, got {len(counts)}"
    assert all(c == args.k for c in counts.values()), f"not all classes have k={args.k}: {counts}"
    print(f"PK composition OK: {args.p} classes x {args.k} samples each")

    grid = args.out / "aug_sample_grid.png"
    save_aug_grid(images, labels, grid, p_show=min(args.p, 6), k=args.k)
    print(f"augmented grid -> {grid}")

    summary = {
        "split": args.split,
        "num_images": len(df),
        "num_classes": num_classes,
        "p": args.p,
        "k": args.k,
        "batch_shape": list(images.shape),
        "batches_per_epoch": len(sampler),
        "pixel_min": round(float(images.min()), 3),
        "pixel_max": round(float(images.max()), 3),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "smoke.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {args.out / 'smoke.json'}")


if __name__ == "__main__":
    main()
