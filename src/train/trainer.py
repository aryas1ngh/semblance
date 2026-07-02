"""Triplet-loss trainer for the SOP embedder."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import PKSampler, SOPDataset, add_contiguous_labels, load_split
from src.data.augmentations import build_eval_transform, build_train_transform
from src.eval import embed_loader, recall_at_k
from src.models.embedder import build_embedder, resolve_device


@dataclass
class TrainConfig:
    """Hyperparameters and paths for a training run."""

    root: Path = Path("data/Stanford_Online_Products")
    embedding_dim: int = 512
    img_size: int = 224
    p: int = 16
    k: int = 4
    lr: float = 1e-4
    epochs: int = 10
    margin: float = 0.2
    batches_per_epoch: int | None = None
    num_workers: int = 4
    seed: int = 42
    subset_classes: int | None = None  # limit train classes (sanity/quick runs)
    eval_classes: int = 200  # test classes sampled for Recall monitoring
    device: str = "auto"  # "auto" | "cpu" | "mps" | "cuda"
    out_dir: Path = field(default_factory=lambda: Path("results/train"))


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _eval_loader(df, root: Path, img_size: int, num_workers: int) -> DataLoader:
    # num_workers=0: eval is small (~1k imgs) and avoids per-epoch worker
    # teardown noise ("can only test a child process") in notebooks.
    ds = SOPDataset(root, df, build_eval_transform(img_size))
    return DataLoader(ds, batch_size=128, num_workers=0)


def train(cfg: TrainConfig) -> dict[str, float]:
    """Train the embedder with triplet loss; return the best metrics."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device.type} | embedding_dim={cfg.embedding_dim} | P={cfg.p} K={cfg.k}")

    # --- train data (optionally restricted to a few classes for sanity) ---
    train_df, _ = add_contiguous_labels(load_split(cfg.root, "train"))
    if cfg.subset_classes is not None:
        train_df = train_df[train_df["label"] < cfg.subset_classes].reset_index(drop=True)
        train_df, _ = add_contiguous_labels(train_df)  # recompact labels
    train_ds = SOPDataset(cfg.root, train_df, build_train_transform(cfg.img_size))
    sampler = PKSampler(
        train_df["label"].to_numpy(), cfg.p, cfg.k, cfg.batches_per_epoch, cfg.seed
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,  # keep workers alive across epochs
    )

    # --- eval data: sanity evaluates on the train subset, else a test-class sample ---
    if cfg.subset_classes is not None:
        eval_df = train_df
    else:
        test_df = load_split(cfg.root, "test")
        classes = np.random.default_rng(cfg.seed).choice(
            test_df["class_id"].unique(), cfg.eval_classes, replace=False
        )
        eval_df = test_df[test_df["class_id"].isin(classes)].reset_index(drop=True)
        eval_df["label"] = eval_df["class_id"]  # class_id works as the grouping label
    eval_loader = _eval_loader(eval_df, cfg.root, cfg.img_size, cfg.num_workers)
    print(f"train: {len(train_ds):,} imgs / {sampler.num_batches} batches | eval: {len(eval_df):,} imgs")

    # --- model, loss, optimizer ---
    model = build_embedder(cfg.embedding_dim).to(device)
    loss_fn = losses.TripletMarginLoss(margin=cfg.margin)
    miner = miners.TripletMarginMiner(margin=cfg.margin, type_of_triplets="semihard")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    metrics_csv = cfg.out_dir / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "recall@1", "recall@10"])

    best_r1 = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sampler.set_epoch(epoch)
        running = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            emb = model(images)
            loss = loss_fn(emb, labels, miner(emb, labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = running / sampler.num_batches

        embs, labs = embed_loader(model, eval_loader, device)
        rec = recall_at_k(embs, labs, (1, 10))
        print(f"epoch {epoch:>3} | loss {avg_loss:.4f} | R@1 {rec[1] * 100:5.1f} | R@10 {rec[10] * 100:5.1f}")
        with metrics_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, round(avg_loss, 4), round(rec[1], 4), round(rec[10], 4)])

        if rec[1] > best_r1:
            best_r1 = rec[1]
            torch.save(model.state_dict(), cfg.out_dir / "best.pt")

    print(f"best Recall@1: {best_r1 * 100:.1f}% -> {cfg.out_dir / 'best.pt'}")
    return {"best_recall@1": best_r1}
