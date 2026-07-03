# Model weights

Trained checkpoints live here but are **not committed** (gitignored, ~94 MB).

To serve the model, download `best.pt` from the Kaggle training run
(`notebooks/kaggle_train.ipynb` → Output panel → `out/best.pt`) and place it at:

```
models/best.pt
```

The index builder (`python -m src.index.build`) and the API (`src/serve/api.py`)
both default to this path.
