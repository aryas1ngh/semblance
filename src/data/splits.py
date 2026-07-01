"""Loading and label preparation for Stanford Online Products (SOP) splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Split name -> the SOP file listing its images.
SPLIT_FILES = {"train": "Ebay_train.txt", "test": "Ebay_test.txt"}


def load_split(root: Path, split: str) -> pd.DataFrame:
    """Load an SOP split into a DataFrame.

    The SOP index files are whitespace-delimited with a header row
    ``image_id class_id super_class_id path``. A readable ``category`` column is
    derived from the image folder (e.g. ``bicycle_final/x.JPG`` -> ``bicycle``).

    Args:
        root: Path to the extracted ``Stanford_Online_Products`` directory.
        split: Either ``"train"`` or ``"test"``.

    Returns:
        DataFrame with columns ``image_id, class_id, super_class_id, path,
        category``.

    Raises:
        ValueError: If ``split`` is not a known split name.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")

    df = pd.read_csv(Path(root) / SPLIT_FILES[split], sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    df["category"] = df["path"].str.split("/").str[0].str.replace("_final", "", regex=False)
    return df


def add_contiguous_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add a contiguous ``label`` column remapping ``class_id`` to ``[0, N)``.

    SOP ``class_id`` values are 1-based and disjoint between the train and test
    splits, so within a split they are not contiguous. Metric-learning samplers
    and heads expect dense integer labels; this maps each distinct ``class_id``
    (in sorted order, for determinism) to ``0..N-1``.

    Args:
        df: A split DataFrame from :func:`load_split`.

    Returns:
        A tuple of ``(df_with_label, num_classes)``. The input is not mutated.
    """
    out = df.copy()
    codes, uniques = pd.factorize(out["class_id"], sort=True)
    out["label"] = codes
    return out, len(uniques)
