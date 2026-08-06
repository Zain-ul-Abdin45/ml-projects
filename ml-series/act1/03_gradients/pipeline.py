"""
Gradients and How Models Actually Learn
Act 1, Article 3 of the ML Series

Companion code for: https://github.com/Zain-ul-Abdin45/ml-projects

Runs standalone. Downloads the real NYC TLC data, reproduces the Article 1
cleaning/feature-engineering pipeline, then implements loss functions,
gradient descent (batch / SGD / mini-batch), and a learning-rate sweep
entirely from scratch — no sklearn model-fitting anywhere in this file.
"""

import numpy as np
import pandas as pd

# ── Pinned data URLs — same as Articles 1 & 2 ──────────────────────────────
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

RNG_SEED = 0
SAMPLE_SIZE = 5000

# The from-scratch GD demos in this article use a 4-feature subset —
# small enough that batch GD, SGD, and mini-batch all converge in seconds,
# and small enough to print a closed-form solution to compare against.
DEMO_FEATURES = ["trip_distance", "fare_amount", "fare_per_mile", "is_rush_hour"]
TARGET = "tip_pct"


# ── Reproduce Article 1 pipeline (same cleaning/engineering as Article 2) ──

def load_raw() -> pd.DataFrame:
    print("Loading TLC data …")
    return pd.read_parquet(DATA_URL)


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["fare_amount"] > 0]
    df = df[df["tip_amount"] >= 0]
    df = df[df["passenger_count"] >= 1]
    df = df[df["passenger_count"] <= 5]
    df = df[df["trip_distance"] > 0]
    df = df[df["trip_distance"] <= 50]
    df = df[df["fare_amount"] >= 3.0]
    df = df[df["fare_amount"] <= 200]

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[df["trip_duration_min"] > 1]
    df = df[df["trip_duration_min"] <= 120]
    df = df[df["RatecodeID"].isin([1, 2])]
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["tpep_pickup_datetime"]
    df["is_rush_hour"] = (
        ((dt.dt.hour >= 7) & (dt.dt.hour <= 9)) |
        ((dt.dt.hour >= 16) & (dt.dt.hour <= 19))
    ).astype(int)
    df["fare_per_mile"]   = (df["fare_amount"] / df["trip_distance"]).clip(upper=50)
    df["is_card_payment"] = (df["payment_type"] == 1).astype(int)
    return df


def build_demo_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Real TLC data, cleaned and engineered exactly as Article 1/2, then
    reduced to the 4-feature demo subset and sampled down to SAMPLE_SIZE
    rows so the from-scratch gradient descent loops run in seconds.

    fare_amount and fare_per_mile are raw dollar values (up to $200 / $50
    per mile) — orders of magnitude larger than a z-scored synthetic
    feature. Feeding that straight into gradient descent blows up at any
    of the learning rates this article compares, so features are
    standardized here, same as the StandardScaler step in Article 2's
    sklearn pipelines. is_rush_hour is already a 0/1 flag."""
    df = load_raw()
    df = clean_trips(df)
    df = engineer_features(df)

    df = df[df["is_card_payment"] == 1].copy()
    df[TARGET] = (df["tip_amount"] / df["fare_amount"] * 100).clip(0, 50)

    df = df[DEMO_FEATURES + [TARGET]].dropna()
    df = df.sample(n=SAMPLE_SIZE, random_state=RNG_SEED).reset_index(drop=True)

    X = df[DEMO_FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std

    print(f"Demo dataset: {len(y):,} real trips, features={DEMO_FEATURES}")
    print(f"Feature means (raw): {np.round(X_mean, 3)}  stds (raw): {np.round(X_std, 3)}")
    return X, y


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(X), 1)), X])


def predict(Xb: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Xb @ w


# ── Loss functions and their gradients ─────────────────────────────────────

def mse_loss(y, pred):
    return float(np.mean((pred - y) ** 2))


def gradient_mse(Xb, y, w):
    pred = predict(Xb, w)
    return (2 / len(y)) * Xb.T @ (pred - y)


def gradient_mae(Xb, y, w):
    pred = predict(Xb, w)
    return (1 / len(y)) * Xb.T @ np.sign(pred - y)


def gradient_huber(Xb, y, w, delta=5.0):
    pred = predict(Xb, w)
    err = pred - y
    grad = np.where(np.abs(err) <= delta, err, delta * np.sign(err))
    return (1 / len(y)) * Xb.T @ grad


def run_gd(grad_fn, Xb, y, lr, epochs):
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        w -= lr * grad_fn(Xb, y, w)
    return w


# ── Section 1: loss functions vs. outliers ──────────────────────────────────

def section_outlier_robustness(X, y):
    """How much do 15 garbage tip_pct values (out of 5,000 rows) move each
    loss function's fitted weights?

    Note: an earlier version of this experiment compared all three losses
    by RMSE on the clean holdout. That's a biased comparison — RMSE is
    literally the quantity MSE minimizes, so any model NOT trained on MSE
    is being judged on a metric it was never optimizing for. The honest,
    metric-independent question is: fit each loss on clean-only data, fit
    it again with the outliers mixed in, and measure how far the learned
    weights moved. That isolates outlier sensitivity from evaluation-metric
    mismatch."""
    print("\n" + "=" * 70)
    print("SECTION 1 — Loss functions vs. injected outliers")
    print("=" * 70)

    Xb = add_bias(X)
    rng = np.random.default_rng(RNG_SEED)
    outlier_idx = rng.choice(len(y), 15, replace=False)
    y_out = y.copy()
    y_out[outlier_idx] = 200  # garbage tip_pct, like a data-entry error

    clean_mask = np.ones(len(y), dtype=bool)
    clean_mask[outlier_idx] = False
    Xb_clean, y_clean = Xb[clean_mask], y[clean_mask]

    # epochs chosen so every loss has actually converged (verified by
    # checking loss/weights stop moving at 2x these epoch counts)
    configs = {
        "MSE":   (gradient_mse,   0.1, 5000),
        "MAE":   (gradient_mae,   0.1, 50000),
        "Huber": (gradient_huber, 0.1, 20000),
    }

    results = {}
    for name, (grad_fn, lr, epochs) in configs.items():
        w_clean = run_gd(grad_fn, Xb_clean, y_clean, lr=lr, epochs=epochs)
        w_out   = run_gd(grad_fn, Xb,       y_out,   lr=lr, epochs=epochs)
        drift = float(np.linalg.norm(w_out - w_clean))
        print(f"{name:6s} w_clean={np.round(w_clean, 3)}")
        print(f"{'':6s} w_out  ={np.round(w_out, 3)}   weight drift (L2): {drift:.4f}")
        results[name] = {"w_clean": w_clean, "w_out": w_out, "drift": drift}

    return results


# ── Section 2: batch vs. SGD vs. mini-batch ─────────────────────────────────

def batch_gd(Xb, y, lr=0.1, epochs=200):
    w = np.zeros(Xb.shape[1])
    losses = []
    for _ in range(epochs):
        grad = gradient_mse(Xb, y, w)
        w -= lr * grad
        losses.append(mse_loss(y, predict(Xb, w)))
    return w, losses


def sgd(Xb, y, lr=0.01, epochs=10, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(Xb.shape[1])
    losses = []
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            xi, yi = Xb[i:i + 1], y[i:i + 1]
            grad = 2 * xi.T @ (predict(xi, w) - yi)
            w -= lr * grad.flatten()
        losses.append(mse_loss(y, predict(Xb, w)))
    return w, losses


def minibatch_gd(Xb, y, lr=0.05, epochs=50, batch_size=64, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(Xb.shape[1])
    losses = []
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = idx[start:start + batch_size]
            grad = gradient_mse(Xb[batch], y[batch], w)
            w -= lr * grad
        losses.append(mse_loss(y, predict(Xb, w)))
    return w, losses


def section_gd_variants(X, y):
    print("\n" + "=" * 70)
    print("SECTION 2 — Batch vs. SGD vs. mini-batch")
    print("=" * 70)

    Xb = add_bias(X)
    w_star, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    print("Closed-form weights (the actual optimum): ", np.round(w_star, 3))

    w_batch, loss_batch = batch_gd(Xb, y, lr=0.1, epochs=200)
    w_sgd,   loss_sgd   = sgd(Xb, y, lr=0.01, epochs=10, seed=RNG_SEED)
    w_mb,    loss_mb    = minibatch_gd(Xb, y, lr=0.05, epochs=50, batch_size=64, seed=RNG_SEED)

    print(f"Batch GD final weights:     {np.round(w_batch, 3)}   final loss: {loss_batch[-1]:.3f}")
    print(f"SGD final weights:          {np.round(w_sgd, 3)}   final loss: {loss_sgd[-1]:.3f}")
    print(f"Mini-batch final weights:   {np.round(w_mb, 3)}   final loss: {loss_mb[-1]:.3f}")

    print("\nBatch GD loss (every 40 of 200 epochs):   ", [round(v, 3) for v in loss_batch[::40]])
    print("SGD loss (every 1 of 10 epochs):           ", [round(v, 3) for v in loss_sgd])
    print("Mini-batch loss (every 10 of 50 epochs):   ", [round(v, 3) for v in loss_mb[::10]])

    return w_star, w_batch, w_sgd, w_mb


# ── Section 3: learning rate sweep ──────────────────────────────────────────

def section_learning_rate(X, y):
    print("\n" + "=" * 70)
    print("SECTION 3 — Learning rate sweep")
    print("=" * 70)

    Xb = add_bias(X)
    for lr in [0.001, 0.01, 0.1, 0.5, 1.0, 1.5]:
        _, losses = batch_gd(Xb, y, lr, epochs=30)
        e5, e30 = losses[4], losses[-1]
        e5_str  = f"{e5:.3f}"   if np.isfinite(e5)  else "inf"
        e30_str = f"{e30:.3f}"  if np.isfinite(e30) else "inf"
        print(f"lr={lr:<6} loss@epoch5={e5_str:>14}   loss@epoch30={e30_str}")


def main():
    X, y = build_demo_dataset()
    section_outlier_robustness(X, y)
    section_gd_variants(X, y)
    section_learning_rate(X, y)
    print("\nDone.")


if __name__ == "__main__":
    main()
