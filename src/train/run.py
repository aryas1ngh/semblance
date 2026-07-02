"""CLI entrypoint for training. Run: python -m src.train.run [--sanity]."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.trainer import TrainConfig, train


def main() -> None:
    """Parse args and launch training."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/Stanford_Online_Products"))
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--p", type=int, default=16, help="classes per batch")
    p.add_argument("--k", type=int, default=4, help="samples per class")
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--batches-per-epoch", type=int, default=None)
    p.add_argument("--eval-classes", type=int, default=200, help="test classes for Recall monitoring")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("results/train"))
    p.add_argument(
        "--sanity",
        action="store_true",
        help="overfit 5 classes to check the loop wires up (expect Recall@1 > 90%)",
    )
    args = p.parse_args()

    cfg = TrainConfig(
        root=args.root,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        lr=args.lr,
        p=args.p,
        k=args.k,
        margin=args.margin,
        batches_per_epoch=args.batches_per_epoch,
        eval_classes=args.eval_classes,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
    )
    if args.sanity:
        cfg.subset_classes = 5
        cfg.p = 5
        cfg.batches_per_epoch = 30
        cfg.epochs = 5
        cfg.device = "cpu"  # MPS lacks cdist_backward for triplet loss
        cfg.out_dir = Path("results/train_sanity")

    train(cfg)


if __name__ == "__main__":
    main()
