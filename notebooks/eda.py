"""Exploratory data analysis for Stanford Online Products (SOP).

Run as a script once the dataset is extracted:

    .venv/bin/python notebooks/eda.py --root data/Stanford_Online_Products

Produces summary stats to stdout and figures under results/eda/. This is EDA
only -- no training logic lives here.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

SPLITS = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}


def load_split(root: Path, filename: str) -> pd.DataFrame:
    """Load an SOP split file into a DataFrame.

    The files are whitespace-delimited with a header row:
    ``image_id class_id super_class_id path``.
    """
    df = pd.read_csv(root / filename, sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    # super_class is encoded in the folder name too; keep the readable label.
    df["category"] = df["path"].str.split("/").str[0].str.replace("_final", "", regex=False)
    return df


def summarize(name: str, df: pd.DataFrame) -> None:
    """Print headline counts for one split."""
    print(f"\n=== {name} ===")
    print(f"  images        : {len(df):,}")
    print(f"  classes       : {df['class_id'].nunique():,}")
    print(f"  super-classes : {df['super_class_id'].nunique()}")
    sizes = df.groupby("class_id").size()
    print(
        f"  images/class  : min={sizes.min()} median={int(sizes.median())} "
        f"max={sizes.max()} mean={sizes.mean():.2f}"
    )


def plot_class_sizes(df: pd.DataFrame, out: Path) -> None:
    """Histogram of images-per-class (the core metric-learning difficulty)."""
    sizes = df.groupby("class_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes, bins=range(1, sizes.max() + 2), edgecolor="black")
    ax.set_xlabel("images per class")
    ax.set_ylabel("number of classes")
    ax.set_title("SOP train: images per product class")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_category_counts(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of images per super-category."""
    counts = df["category"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, edgecolor="black")
    ax.set_ylabel("images")
    ax.set_title("SOP train: images per category")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def image_size_stats(root: Path, df: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    """Measure width/height on a random sample of images."""
    rng = random.Random(seed)
    paths = rng.sample(list(df["path"]), k=min(sample, len(df)))
    rows = []
    for rel in paths:
        with Image.open(root / rel) as im:
            rows.append({"width": im.width, "height": im.height, "mode": im.mode})
    return pd.DataFrame(rows)


def sample_grid(root: Path, df: pd.DataFrame, out: Path, seed: int, n: int = 12) -> None:
    """Save a grid of one random image per category."""
    rng = random.Random(seed)
    cats = sorted(df["category"].unique())[:n]
    cols = 4
    rows = (len(cats) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    for ax, cat in zip(axes.ravel(), cats, strict=False):
        rel = rng.choice(list(df.loc[df["category"] == cat, "path"]))
        with Image.open(root / rel) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(cat, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(cats) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/Stanford_Online_Products"))
    parser.add_argument("--out", type=Path, default=Path("results/eda"))
    parser.add_argument("--sample", type=int, default=1000, help="images to measure size on")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    splits = {name: load_split(args.root, fn) for name, fn in SPLITS.items()}
    for name, df in splits.items():
        summarize(name, df)

    # Confirm the paper's disjoint class split: no product id shared across splits.
    overlap = set(splits["train"]["class_id"]) & set(splits["test"]["class_id"])
    print(f"\nclass_id overlap train/test: {len(overlap)} (expected 0)")

    train = splits["train"]
    sizes = image_size_stats(args.root, train, args.sample, args.seed)
    print(f"\n=== image sizes (sample of {len(sizes)}) ===")
    print(sizes[["width", "height"]].describe().round(1).to_string())
    print("  modes:", sizes["mode"].value_counts().to_dict())

    plot_class_sizes(train, args.out / "class_sizes.png")
    plot_category_counts(train, args.out / "category_counts.png")
    sample_grid(args.root, train, args.out / "sample_grid.png", args.seed)
    print(f"\nfigures written to {args.out}/")


if __name__ == "__main__":
    main()
