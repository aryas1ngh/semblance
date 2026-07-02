"""Embedding extraction and leave-one-out Recall@K."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@torch.no_grad()
def embed_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return (embeddings, labels) for every sample in ``loader``."""
    model.eval()
    embs, labels = [], []
    for batch in loader:
        images, targets = batch[0], batch[1]
        embs.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(embs), torch.cat(labels)


def recall_at_k(embeddings: Tensor, labels: Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    """Leave-one-out Recall@K via cosine similarity (embeddings assumed normalized)."""
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-1.0)
    top = sims.topk(max(ks), dim=1).indices
    hits = labels[top] == labels.unsqueeze(1)
    return {k: hits[:, :k].any(dim=1).float().mean().item() for k in ks}
