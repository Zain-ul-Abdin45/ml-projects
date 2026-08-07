# Act 1 / Article 1 — Data From Scratch

Companion code for **What "Learning from Data" Actually Means — Before Any Model**.

Article link: [medium.com/@zainkhoso45/what-learning-from-data-actually-means-before-any-model-84acc939c05b](https://medium.com/@zainkhoso45/what-learning-from-data-actually-means-before-any-model-84acc939c05b)

**Series:** Article 1 (this one) → Article 2, Loss Functions (`../02_loss_functions`, unpublished) → [Article 3, Gradients and How Models Actually Learn](https://medium.com/@zainkhoso45/gradients-and-how-models-actually-learn-4791d031f65a) (`../03_gradients`)

## What this folder contains

| File | Description |
|---|---|
| `pipeline.py` | Standalone script — loads TLC data, cleans it with intent, engineers features, and produces `X_train / y_train / X_test / y_test`. No `model.fit()`. |
| `notebook.ipynb` | Same pipeline in a runnable notebook with inline output |
| `architecture.mmd` | Mermaid diagram of every step from raw TLC parquet to the final split |
| `requirements.txt` | Python dependencies |

## Run it

```bash
pip install -r requirements.txt
python pipeline.py
```

Downloads ~250 MB of TLC parquet data on first run. Everything else runs locally.

## Pinned data

```python
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"
```

January 2024 — pinned so the notebook and article stay in sync.

## What this does NOT do

No `model.fit()`. The article is about what happens before that line — and this code ends where the split is ready.
