"""Prototype: off-the-shelf ResNet50 as a visual product search baseline.

Embeds a gallery of SOP test-split products with an ImageNet-pretrained ResNet50
(no metric-learning fine-tuning), then measures leave-one-out Recall@K and saves
a qualitative query -> top-K retrieval grid. This is the baseline our trained
model must beat; it is throwaway prototype code that later refactors into
``src/models`` (backbone), ``src/index`` (search), and ``src/eval`` (metrics).

    .venv/bin/python prototype_search.py --n-classes 600
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


class ImagePathDataset(Dataset[tuple[Tensor, int]]):
    """Loads images from relative paths and applies the backbone's transform."""

    def __init__(self, root: Path, paths: list[str], transform) -> None:  # noqa: ANN001
        self.root = root
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        with Image.open(self.root / self.paths[idx]) as im:
            img = im.convert("RGB")  # SOP has a few grayscale images
        return self.transform(img), idx


def load_test_split(root: Path) -> pd.DataFrame:
    """Read Ebay_test.txt into a DataFrame with a readable category column."""
    df = pd.read_csv(root / "Ebay_test.txt", sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    df["category"] = df["path"].str.split("/").str[0].str.replace("_final", "", regex=False)
    return df


def sample_gallery(df: pd.DataFrame, n_classes: int, seed: int) -> pd.DataFrame:
    """Sample all images from a random subset of classes.

    Sampling whole classes (not random images) guarantees every image has at
    least one same-class positive present, so Recall@K is well-defined.
    """
    rng = random.Random(seed)
    classes = sorted(df["class_id"].unique())
    chosen = set(rng.sample(classes, k=min(n_classes, len(classes))))
    return df[df["class_id"].isin(chosen)].reset_index(drop=True)


@torch.no_grad()
def embed(
    model: torch.nn.Module,
    loader: DataLoader[tuple[Tensor, int]],
    n: int,
    dim: int,
    device: torch.device,
) -> Tensor:
    """Run the backbone over the loader and return L2-normalized embeddings."""
    out = torch.empty((n, dim), dtype=torch.float32)
    for batch, idx in tqdm(loader, desc="embedding", unit="batch"):
        feats = model(batch.to(device)).cpu()
        out[idx] = feats
    return torch.nn.functional.normalize(out, dim=1)


def recall_at_k(emb: Tensor, labels: Tensor, ks: tuple[int, ...]) -> dict[int, float]:
    """Leave-one-out Recall@K via cosine similarity (embeddings are normalized)."""
    sims = emb @ emb.T
    sims.fill_diagonal_(-1.0)  # exclude the query itself
    max_k = max(ks)
    top = sims.topk(max_k, dim=1).indices  # (N, max_k)
    top_labels = labels[top]
    hits = top_labels == labels.unsqueeze(1)  # (N, max_k) True where same class
    return {k: (hits[:, :k].any(dim=1).float().mean().item()) for k in ks}


def save_query_grid(
    root: Path,
    gallery: pd.DataFrame,
    emb: Tensor,
    labels: Tensor,
    out: Path,
    seed: int,
    n_queries: int = 4,
    k: int = 5,
) -> None:
    """Save a grid: each row is a query image followed by its top-K neighbors."""
    sims = emb @ emb.T
    sims.fill_diagonal_(-1.0)
    rng = random.Random(seed)
    queries = rng.sample(range(len(gallery)), k=n_queries)
    fig, axes = plt.subplots(n_queries, k + 1, figsize=((k + 1) * 1.9, n_queries * 2.1))
    for r, q in enumerate(queries):
        neigh_sim, neigh = sims[q].topk(k)
        cells = [(q, None, True)] + [
            (int(n), float(s), bool(labels[n] == labels[q]))
            for n, s in zip(neigh, neigh_sim)
        ]
        for c, (idx, sim, same) in enumerate(cells):
            ax = axes[r, c]
            with Image.open(root / gallery.loc[idx, "path"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("query", fontsize=9)
                for sp in ax.spines.values():
                    sp.set_color("black")
                    sp.set_linewidth(2)
            else:
                color = "green" if same else "red"
                ax.set_title(f"{sim:.2f}", fontsize=9, color=color)
                for sp in ax.spines.values():
                    sp.set_color(color)
                    sp.set_linewidth(2.5)
    fig.suptitle("Pretrained ResNet50 retrieval — green = same product, red = not", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/Stanford_Online_Products"))
    parser.add_argument("--n-classes", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/prototype"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # expose the 2048-d pooled features
    model.eval().to(device)

    df = load_test_split(args.root)
    gallery = sample_gallery(df, args.n_classes, args.seed)
    labels = torch.tensor(gallery["class_id"].to_numpy())
    print(
        f"gallery: {len(gallery):,} images across {gallery['class_id'].nunique():,} "
        f"classes | device={device.type}"
    )

    ds = ImagePathDataset(args.root, list(gallery["path"]), weights.transforms())
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )

    t0 = time.perf_counter()
    emb = embed(model, loader, len(gallery), 2048, device)
    print(f"embedded in {time.perf_counter() - t0:.1f}s")

    ks = (1, 5, 10)
    recalls = recall_at_k(emb, labels, ks)
    print("\n=== no-training baseline (ImageNet ResNet50) ===")
    for k in ks:
        print(f"  Recall@{k:<2}: {recalls[k] * 100:5.1f}%")

    grid_path = args.out / "query_topk.png"
    save_query_grid(args.root, gallery, emb, labels, grid_path, args.seed)
    print(f"\nqualitative grid -> {grid_path}")

    metrics = {"n_classes": args.n_classes, "n_images": len(gallery), "seed": args.seed}
    metrics.update({f"recall@{k}": recalls[k] for k in ks})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(args.out / "baseline_metrics.json", indent=2)
    print(f"metrics -> {args.out / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
