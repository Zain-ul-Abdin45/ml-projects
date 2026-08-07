# ml-series

An ML series told through one real dataset (NYC TLC taxi trips, January 2024) end to end — cleaning, target selection, loss functions, gradient descent, and eventually a production model — with every number in every article coming from code that actually ran, never a synthetic stand-in.

## Act 1

| # | Article | Code | Status |
|---|---|---|---|
| 1 | [What "Learning from Data" Actually Means — Before Any Model](https://medium.com/@zainkhoso45/what-learning-from-data-actually-means-before-any-model-84acc939c05b) | [`act1/01_data_from_scratch`](act1/01_data_from_scratch) | Published |
| 2 | Loss Functions — What Your Model Is Actually Optimizing | [`act1/02_loss_functions`](act1/02_loss_functions) | Unpublished |
| 3 | [Gradients and How Models Actually Learn](https://medium.com/@zainkhoso45/gradients-and-how-models-actually-learn-4791d031f65a) | [`act1/03_gradients`](act1/03_gradients) | Published |

Article 3 published ahead of Article 2 and picks up directly from Article 1 — it covers loss functions from scratch as part of its own scope, so Article 2 may end up folded into it rather than published separately.

Each article folder is self-contained: `pipeline.py` (standalone script), `notebook.ipynb` (same code, narrative cells), `README.md`, `architecture.mmd` (Mermaid diagram of that article's core mechanism), and `requirements.txt`. All three pull the same pinned data:

```python
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
```
