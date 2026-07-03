"""Build the FAISS gallery index from a trained checkpoint.

Embeds the SOP test split, saves an inner-product FAISS index + aligned
metadata, and reports honest full-test Recall@1/@10.

Recall is computed with a chunked torch matmul (not faiss) to keep memory bounded
and avoid the torch/faiss OpenMP clash on macOS; faiss is used only to store and
serve the index.

Run: python -m src.index.build --ckpt models/best.pt
"""

from __future__ import annotations

import os

# torch, faiss, and opencv each ship an OpenMP runtime; on macOS the three copies
# clash and crash. Allow the duplicate and pin a single OMP thread pool. Embedding
# runs on MPS/CUDA so this doesn't slow the GPU path. Harmless in the Linux image.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import faiss
import torch
from torch.utils.data import DataLoader

from src.data import SOPDataset, add_contiguous_labels, load_split
from src.data.augmentations import build_eval_transform
from src.eval import embed_loader
from src.models.embedder import build_embedder, resolve_device

faiss.omp_set_num_threads(1)


def recall_full(embs, labels, ks=(1, 10), chunk=1024) -> dict[int, float]:
    """Leave-one-out Recall@K via chunked cosine similarity on ``embs``' device."""
    n = embs.shape[0]
    hits = {k: 0 for k in ks}
    for s in range(0, n, chunk):
        q = embs[s : s + chunk]
        sims = q @ embs.T  # (chunk, N), cosine since embeddings are normalized
        rows = torch.arange(q.shape[0], device=sims.device)
        sims[rows, rows + s] = -1.0  # drop self-match
        top = sims.topk(max(ks), dim=1).indices
        match = labels[top] == labels[s : s + q.shape[0], None]
        for k in ks:
            hits[k] += int(match[:, :k].any(1).sum())
    return {k: hits[k] / n for k in ks}


def build(cfg: argparse.Namespace) -> None:
    device = resolve_device(cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    df, _ = add_contiguous_labels(load_split(cfg.root, "test"))
    if cfg.limit:
        df = df.head(cfg.limit).reset_index(drop=True)
    ds = SOPDataset(cfg.root, df, build_eval_transform(cfg.img_size))
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = build_embedder(cfg.embedding_dim).to(device)
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device))
    print(f"loaded {cfg.ckpt} | device={device.type} | gallery={len(ds):,} imgs")

    embs, labels = embed_loader(model, loader, device)  # rows align with df order
    rec = recall_full(embs.to(device), labels.to(device), (1, 10))
    print(f"full-test Recall@1 {rec[1] * 100:.1f} | Recall@10 {rec[10] * 100:.1f}")

    embs_np = embs.numpy().astype("float32")
    index = faiss.IndexFlatIP(embs_np.shape[1])  # inner product == cosine (normalized)
    index.add(embs_np)
    faiss.write_index(index, str(cfg.out_dir / "gallery.faiss"))
    df[["path", "class_id", "super_class_id", "category"]].to_csv(
        cfg.out_dir / "gallery_meta.csv", index=False
    )
    (cfg.out_dir / "metrics.json").write_text(
        json.dumps({"recall@1": rec[1], "recall@10": rec[10], "gallery_size": len(ds)}, indent=2)
    )
    print(f"wrote index + metadata -> {cfg.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("models/best.pt"))
    p.add_argument("--root", type=Path, default=Path("data/Stanford_Online_Products"))
    p.add_argument("--out-dir", type=Path, default=Path("results/index"))
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap gallery size (0 = full test set)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    build(p.parse_args())


if __name__ == "__main__":
    main()
