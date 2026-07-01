"""PK batch sampler for metric learning."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class PKSampler:
    """Yield batches of ``P`` classes with ``K`` samples each (batch = P*K).

    Metric-learning losses (triplet, contrastive) need several examples per
    class in every batch so that positive pairs and hard negatives can be mined.
    This sampler draws ``P`` distinct classes per batch and ``K`` indices from
    each. When a class has fewer than ``K`` images (SOP has classes with only 2),
    its indices are drawn **with replacement** so the batch shape stays fixed.

    Use as ``DataLoader(dataset, batch_sampler=PKSampler(...))``.

    Args:
        labels: Per-sample integer class labels, aligned with dataset indices.
        p: Number of distinct classes per batch.
        k: Number of samples per class per batch.
        num_batches: Batches per epoch. Defaults to ``len(labels) // (p*k)``.
        seed: Base RNG seed; each epoch reseeds deterministically from it.

    Raises:
        ValueError: If ``p`` exceeds the number of available classes.
    """

    def __init__(
        self,
        labels: np.ndarray,
        p: int,
        k: int,
        num_batches: int | None = None,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels)
        self.p = p
        self.k = k
        self.seed = seed
        self.epoch = 0

        self._class_to_indices: dict[int, np.ndarray] = {
            int(c): np.flatnonzero(self.labels == c) for c in np.unique(self.labels)
        }
        self._classes = np.array(sorted(self._class_to_indices))
        if p > len(self._classes):
            raise ValueError(f"p={p} exceeds number of classes ({len(self._classes)})")

        self.num_batches = num_batches if num_batches is not None else len(self.labels) // (p * k)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so shuffling differs across epochs but stays reproducible."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            classes = rng.choice(self._classes, size=self.p, replace=False)
            batch: list[int] = []
            for c in classes:
                pool = self._class_to_indices[int(c)]
                replace = len(pool) < self.k
                chosen = rng.choice(pool, size=self.k, replace=replace)
                batch.extend(int(i) for i in chosen)
            yield batch
