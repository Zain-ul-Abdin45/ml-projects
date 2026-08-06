# Act 1 / Article 3 — Gradients and How Models Actually Learn

Companion code for **Gradients and How Models Actually Learn**.

Article link: *(add Medium/Substack URL when published)*

## What this folder contains

| File | Description |
|---|---|
| `pipeline.py` | Standalone script — loads real TLC data, reproduces Article 1/2 cleaning, then implements loss functions, batch/SGD/mini-batch gradient descent, a learning-rate sweep, and the two visuals below, entirely from scratch (no `model.fit()` anywhere) |
| `notebook.ipynb` | Same code in a runnable notebook with inline output and narrative cells |
| `architecture.mmd` | Mermaid diagram of the three gradient descent variants and what differs between them |
| `requirements.txt` | Python dependencies — `pandas` / `numpy` / `pyarrow` / `matplotlib`, no `sklearn` |
| `output/loss_landscape.png` | The real loss surface batch GD descends (a 2D slice over the `trip_distance` / `fare_amount` weights, computed analytically), with the actual 200-step batch GD path drawn on top |
| `output/learning_rate_behavior.png` | The five learning-rate runs from the sweep, plotted — zoomed to the first 10 epochs next to the full 30-epoch blowup |

## Run it

```bash
pip install -r requirements.txt
python pipeline.py
```

Downloads ~50 MB of TLC parquet data on first run. Everything else runs locally in seconds — the from-scratch gradient descent loops run on a 5,000-row, 4-feature real subsample (`trip_distance`, `fare_amount`, `fare_per_mile`, `is_rush_hour` → `tip_pct`), not the full 1.6M-row training set. That's a deliberate scope choice: this article is about the mechanics of gradients and learning rates, not about building the production tip% model — that's the sklearn `SGDRegressor` version, saved for the next article.

## What it demonstrates

- **Outlier sensitivity, measured correctly** — fits MSE, MAE, and Huber both on clean data and on data with 15 injected garbage `tip_pct` values, then compares the *weight drift* between the two fits for each loss. (Comparing RMSE-on-clean-data across all three, which an earlier version of this experiment did, structurally favors MSE since RMSE is literally what MSE minimizes — see the comment in `section_outlier_robustness()`.)
- **Batch vs. SGD vs. mini-batch gradient descent**, implemented from scratch, all three converging toward the same closed-form (`np.linalg.lstsq`) optimum from different paths and at different costs per update.
- **A learning-rate sweep** across six values, including the real divergence trap: `lr=0.5` has a *lower* loss than `lr=0.1` for the first four epochs, then reverses and diverges to over 10^8 by epoch 30.
- **Why there's no "local minimum" trap to worry about here** — MSE is an exact quadratic in the weights for plain linear regression, so the loss surface is provably one convex bowl. `loss_landscape.png` shows that surface directly, sliced to the two most-correlated weights, instead of asserting it.

## Pinned data

```python
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
```

January 2024 — same file as Articles 1 & 2, so all three stay in sync.

## Note on feature scaling

`fare_amount` and `fare_per_mile` are raw dollar values (up to $200 / $50 per mile) — much larger scale than a synthetic, pre-normalized feature. Feeding that straight into gradient descent blows up at every learning rate this article tests. Features are standardized (zero mean, unit variance) before the from-scratch GD loops, the same role `StandardScaler` plays in Article 2's sklearn pipelines.
