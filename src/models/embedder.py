"""ResNet50 backbone with a linear embedding head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50


class Embedder(nn.Module):
    """ResNet50 -> Linear head producing (optionally L2-normalized) embeddings."""

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, normalize: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.backbone = backbone
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        emb = self.backbone(x)
        return nn.functional.normalize(emb, dim=1) if self.normalize else emb


def build_embedder(embedding_dim: int = 512, pretrained: bool = True) -> Embedder:
    """Construct an :class:`Embedder`."""
    return Embedder(embedding_dim=embedding_dim, pretrained=pretrained)


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device; "auto" picks cuda, then mps, then cpu.

    Note: triplet loss needs ``cdist_backward``, unimplemented on MPS, so local
    training should use "cpu" (real training runs on CUDA/Kaggle).
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
