# Semblance: Visual Product Search Engine

Upload a product image, get back the most visually similar products from an
indexed gallery, with similarity scores. A metric-learning retrieval system
built on the **Stanford Online Products (SOP)** dataset.

> **Status: in progress.** Data pipeline explored and a no-training baseline is
> in place. Training, indexing, API, and deployment are upcoming. See the
> [roadmap](#roadmap).

---

## Why this is a real retrieval problem

SOP is **open-set**: the 11,316 test products share **zero classes** with the
11,318 train products, so a classifier can't work, the system has to learn an
embedding space where visually similar items land close together and that
generalizes to products never seen in training. That's why the approach is
**metric learning** (triplet / ArcFace), evaluated with **Recall@K**.

Each product has only **2–12 photos** (median 4), which drives the sampling and
loss choices. Full data exploration lives in
[`notebooks/eda.py`](notebooks/eda.py) and
[`notebooks/deep_eda.ipynb`](notebooks/deep_eda.ipynb).

## Targets

| Metric | Target | Note |
| --- | --- | --- |
| Recall@1  | ≥ 65% | on the SOP test split |
| Recall@10 | ≥ 85% | on the SOP test split |

Published SOP SOTA is ~85% R@1; we aim for a credible, defensible number.

## Baseline (no training)

An off-the-shelf ImageNet ResNet50 (2048-d pooled features, cosine similarity),
**no fine-tuning**, measured as the baseline our trained model must beat:

```bash
.venv/bin/python prototype_search.py --n-classes 600
```

| Metric | 600-class subset (3,232 imgs) |
| --- | --- |
| Recall@1  | 71.2% |
| Recall@5  | 83.1% |
| Recall@10 | 86.8% |

> ⚠️ These are on a **subset** (fewer distractors → optimistic). A full
> 60k-image test-set baseline is lower (pretrained ResNet50 is typically
> ~45–50% R@1 on full SOP) and is the honest number to beat. _Full-set baseline
> pending._

Qualitatively (`results/prototype/query_topk.png`), pretrained features nail
distinctive objects but confuse **same-category, different-product** items and
lean heavily on **colour/texture**, the exact gap metric learning closes.

## Planned architecture

```
image ──▶ backbone (ResNet50) ──▶ embedding head ──▶ L2-normalized vector
                                                            │
                                                            ▼
                                             vector DB (Qdrant / FAISS)
                                                            │
   query image ──▶ same embedder ──▶ top-K nearest ────────┘──▶ results + scores
```

Serving: **FastAPI** `POST /search` + **Streamlit** UI, orchestrated with
**docker compose**; lightweight **FAISS** build for a Hugging Face Spaces demo.

## Repo structure

Current (what exists today):

```
data/                 SOP dataset: not committed (see data/README.md)
notebooks/            EDA only: eda.py, deep_eda.ipynb
results/              committed figures + metrics
prototype_search.py   no-training baseline (refactors into src/ later)
```

Planned as later phases land (`src/` with `data/`, `models/`, `train/`, `eval/`,
`index/`, `serve/`, `frontend/`; `configs/` for Hydra; `tests/`; `docker/`).
Scaffolding is added when a phase actually needs it, not before.

## Quickstart

```bash
# 1. Python 3.11 virtualenv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Baseline prototype needs only a few packages
pip install torch torchvision numpy pillow pandas matplotlib tqdm

# 3. Fetch the dataset (see data/README.md)
#    ~3 GB download, ~6 GB extracted

# 4. Explore the data / run the baseline
python notebooks/eda.py
python prototype_search.py --n-classes 600
```

## Dataset

Stanford Online Products: Song et al., *Deep Metric Learning via Lifted
Structured Feature Embedding*, CVPR 2016. Standard split, unchanged. See
[`data/README.md`](data/README.md).

## Roadmap

- [x] Data pipeline & EDA
- [x] No-training baseline (pretrained ResNet50 retrieval)
- [ ] Training (triplet, then ArcFace) with W&B tracking
- [ ] Gallery indexing (Qdrant + FAISS)
- [ ] FastAPI `/search` + Streamlit UI, `docker compose up`
- [ ] Evaluation, embedding-space viz, failure-case analysis
- [ ] Hugging Face Spaces deploy
