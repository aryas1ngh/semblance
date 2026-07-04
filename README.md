# Semblance: Visual Product Search Engine

Upload a product image, get back the most visually similar products from an
indexed gallery, with similarity scores. A metric-learning retrieval system
built on the **Stanford Online Products (SOP)** dataset.

> **Status: in progress.** Data pipeline, trained metric-learning model
> (Recall@1 90% on the eval subset), FAISS gallery index, and a FastAPI +
> Streamlit search service are in place. Deployment is upcoming. See the
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
models/               trained weights: not committed (see models/README.md)
notebooks/            EDA + Kaggle training notebook
src/data/             dataset, PK sampler, augmentations, split loading
src/models/           ResNet50 embedder
src/train/            triplet-loss trainer + CLI
src/eval/             Recall@K
src/index/            build the FAISS gallery index
src/serve/            FastAPI /search service
app/                  Streamlit UI
Dockerfile, docker-compose.yml   run API + UI together
results/              committed figures + metrics
prototype_search.py   no-training baseline (throwaway)
```

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

# 5. Verify the training data pipeline (PK batch + augmentations)
pip install albumentations
python -m src.data.smoke --p 8 --k 4
```

## Search service

With a trained checkpoint at `models/best.pt` and a built index at
`results/index/` (see below), one command brings up the API and UI together:

```bash
docker compose up --build
#   UI   -> http://localhost:8501
#   API  -> http://localhost:8000/docs
```

Both services run from a single CPU-only image; the checkpoint, FAISS index, and
image data are mounted **read-only** at runtime rather than baked in, so the
image stays small and the artifacts stay out of the build context.

<details>
<summary>Build the index / run without Docker</summary>

```bash
# 1. Build the FAISS gallery index (embeds the test split, prints full-test Recall@K)
python -m src.index.build --ckpt models/best.pt

# 2. Start the API (terminal 1)
CKPT=models/best.pt INDEX_DIR=results/index DATA_ROOT=data/Stanford_Online_Products \
  uvicorn src.serve.api:app --port 8000
#    docs -> http://localhost:8000/docs

# 3. Start the UI (terminal 2), then open http://localhost:8501
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```
</details>

The API embeds an uploaded image with the trained model, does a cosine
nearest-neighbour lookup over the gallery with **FAISS** (`IndexFlatIP` on
L2-normalized vectors — exact search; swap to IVF/HNSW or a vector DB like
Qdrant at scale), and returns the top-K products with similarity scores.

On the full 60k test gallery the trained model reaches **Recall@1 63.9% /
Recall@10 81.3%** (vs ~45–50% R@1 for the pretrained baseline).

## Dataset

Stanford Online Products: Song et al., *Deep Metric Learning via Lifted
Structured Feature Embedding*, CVPR 2016. Standard split, unchanged. See
[`data/README.md`](data/README.md).

## Roadmap

- [x] Data pipeline & EDA
- [x] No-training baseline (pretrained ResNet50 retrieval)
- [x] Triplet-loss trainer (`src/train`, `src/models`, `src/eval`) + Kaggle notebook — Recall@1 90% on eval subset
- [x] FAISS gallery index (`src/index`) + full-test Recall@K (R@1 63.9%)
- [x] FastAPI `/search` + Streamlit UI
- [x] `docker compose up` one-command run
- [ ] ArcFace head + W&B tracking
- [ ] Evaluation, embedding-space viz, failure-case analysis
- [ ] Hugging Face Spaces deploy
