"""
Gradients and How Models Actually Learn
Act 1, Article 3 of the ML Series

Companion code for: https://github.com/Zain-ul-Abdin45/ml-projects

Runs standalone. Downloads the real NYC TLC data, reproduces the Article 1
cleaning/feature-engineering pipeline, then implements loss functions,
gradient descent (batch / SGD / mini-batch), and a learning-rate sweep
entirely from scratch — no sklearn model-fitting anywhere in this file.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Pinned data URLs — same as Articles 1 & 2 ──────────────────────────────
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"

RNG_SEED = 0
SAMPLE_SIZE = 5000
OUTPUT_DIR = "output"

# Same categorical colors Article 2 used for MSE/MAE/Huber — kept consistent
# across the series rather than introducing a new palette per article.
BLUE, ORANGE, GREEN, RED = "#2980b9", "#e67e22", "#27ae60", "#e34948"
# Sequential blue ramp (light → dark) for magnitude/loss-surface encoding.
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

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


# ── Visuals: the loss surface, and what different learning rates do to it ──

def batch_gd_path(Xb, y, lr=0.1, epochs=200):
    """Same update rule as batch_gd, but returns every intermediate weight
    vector instead of just the final one — for plotting the descent path."""
    w = np.zeros(Xb.shape[1])
    history = [w.copy()]
    for _ in range(epochs):
        w = w - lr * gradient_mse(Xb, y, w)
        history.append(w.copy())
    return np.array(history)


def loss_surface(Xb, y, w_fixed, i, j, w1_range, w2_range):
    """MSE as a function of two weights (i, j), holding every other weight
    fixed at w_fixed. For linear regression, MSE is an exact quadratic in
    the weights, so the surface is computed analytically instead of by
    brute-force re-evaluating predict() at every grid point:

        error(w1, w2) = r0 + Xi*w1 + Xj*w2          where r0 = base_pred - y
        MSE = mean(r0^2) + 2*w1*mean(r0*Xi) + 2*w2*mean(r0*Xj)
              + w1^2*mean(Xi^2) + w2^2*mean(Xj^2) + 2*w1*w2*mean(Xi*Xj)

    That's also the honest reason this surface has exactly one minimum:
    a quadratic bowl in two variables has no separate local minima, local
    maxima, or saddle points to fall into — that's a property of MSE +
    linear regression, not of gradient descent."""
    Xi, Xj = Xb[:, i], Xb[:, j]
    base = Xb @ w_fixed - Xi * w_fixed[i] - Xj * w_fixed[j]
    r0 = base - y

    c0  = np.mean(r0 ** 2)
    c1  = 2 * np.mean(r0 * Xi)
    c2  = 2 * np.mean(r0 * Xj)
    c11 = np.mean(Xi ** 2)
    c22 = np.mean(Xj ** 2)
    c12 = 2 * np.mean(Xi * Xj)

    W1, W2 = np.meshgrid(w1_range, w2_range)
    Loss = c0 + c1 * W1 + c2 * W2 + c11 * W1 ** 2 + c22 * W2 ** 2 + c12 * W1 * W2
    return W1, W2, Loss


def plot_loss_landscape(X, y, w_star, save_path=None):
    """The loss surface batch GD actually walks across, sliced to two
    correlated weights (trip_distance, fare_amount — the pair Section 2
    calls out as a long, shallow valley), with the real batch GD path
    overlaid."""
    Xb = add_bias(X)
    i, j = 1, 2  # trip_distance weight, fare_amount weight
    path = batch_gd_path(Xb, y, lr=0.1, epochs=200)

    # Frame the grid around whichever is wider — the GD path (starts at 0)
    # or the optimum — with padding, so both are visible with real margin.
    pad = 2.5
    w1_lo, w1_hi = min(0, w_star[i]) - pad, max(0, w_star[i]) + pad
    w2_lo, w2_hi = min(0, w_star[j]) - pad, max(0, w_star[j]) + pad
    w1_range = np.linspace(w1_lo, w1_hi, 200)
    w2_range = np.linspace(w2_lo, w2_hi, 200)
    W1, W2, Loss = loss_surface(Xb, y, w_star, i, j, w1_range, w2_range)

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cf = ax.contourf(W1, W2, Loss, levels=30, cmap=cmap)
    lines = ax.contour(W1, W2, Loss, levels=8, colors="white", linewidths=0.6, alpha=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt="%.0f")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("MSE loss (higher = worse)", color="#52514e")

    ax.plot(path[:, i], path[:, j], color=ORANGE, lw=2, zorder=4,
             label="Batch GD path (200 steps)")
    ax.scatter(path[0, i], path[0, j], color=ORANGE, edgecolor="white",
               s=50, zorder=5, marker="s", label="Start (w = 0)")
    ax.scatter(w_star[i], w_star[j], color="white", edgecolor="black",
               s=140, zorder=5, marker="*", label="Closed-form optimum")

    ax.set_xlabel("trip_distance weight")
    ax.set_ylabel("fare_amount weight")
    ax.set_title("The loss surface batch GD descends\none convex valley — no separate local minima or maxima to fall into",
                 fontsize=11)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.close(fig)


def plot_learning_rate_behavior(X, y, save_path=None):
    """Real loss-vs-epoch curves for five learning rates on the same
    problem. One log-scale axis across all 30 epochs makes lr=0.5 and
    lr=1.0's eventual blowup unmissable, but it also flattens the three
    well-behaved rates into an unreadable band and hides the thing Section
    3 of the article is actually about — that lr=0.5 briefly *beats*
    lr=0.1 in the first four epochs. So: two panels, same data, two zoom
    levels."""
    Xb = add_bias(X)
    lr_colors = [
        (0.001, BLUE_SEQ[1]),
        (0.01,  BLUE_SEQ[3]),
        (0.1,   BLUE),
        (0.5,   ORANGE),
        (1.0,   RED),
    ]
    all_losses = {lr: batch_gd(Xb, y, lr, epochs=30)[1] for lr, _ in lr_colors}

    fig, (ax_zoom, ax_full) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left: first 10 epochs, where lr=0.5 looks like the fast learner ──
    for lr, color in lr_colors:
        if lr == 1.0:
            continue  # already off-scale within a couple of epochs
        losses = all_losses[lr][:10]
        epochs = np.arange(1, len(losses) + 1)
        ax_zoom.plot(epochs, losses, color=color, lw=2.2, marker="o", markersize=4)
        ax_zoom.annotate(f"lr={lr}", xy=(epochs[-1], losses[-1]),
                          xytext=(6, 0), textcoords="offset points",
                          va="center", fontsize=9, color=color, fontweight="bold")
    ax_zoom.axvline(5, color="#898781", lw=1, linestyle=":")
    ax_zoom.text(5.15, 0.95, "epoch 5\ncrossover", transform=ax_zoom.get_xaxis_transform(),
                 ha="left", va="top", fontsize=8, color="#52514e")
    ax_zoom.set_yscale("log")
    ax_zoom.set_xlabel("epoch")
    ax_zoom.set_ylabel("MSE loss (log scale)")
    ax_zoom.set_title("First 10 epochs — lr=0.5 leads, then reverses", fontsize=10)
    ax_zoom.grid(True, which="both", axis="y", color="#e1e0d9", linewidth=0.6)
    ax_zoom.set_axisbelow(True)

    # ── Right: all 30 epochs, full blowup ──
    for lr, color in lr_colors:
        losses = all_losses[lr]
        epochs = np.arange(1, len(losses) + 1)
        ax_full.plot(epochs, losses, color=color, lw=2.2, marker="o", markersize=3)
        if lr in (0.5, 1.0):
            ax_full.annotate(f"lr={lr}", xy=(epochs[-1], losses[-1]),
                              xytext=(6, 0), textcoords="offset points",
                              va="center", fontsize=9, color=color, fontweight="bold")
    ax_full.annotate("lr = 0.001 / 0.01 / 0.1\n(converge, barely visible at this scale)",
                      xy=(4, all_losses[0.1][3]), xytext=(15, 55),
                      textcoords="offset points", fontsize=8, color="#52514e",
                      arrowprops=dict(arrowstyle="-", color="#898781", lw=0.8))
    ax_full.set_yscale("log")
    ax_full.set_xlabel("epoch")
    ax_full.set_ylabel("MSE loss (log scale)")
    ax_full.set_title("All 30 epochs — same two rates diverge past 10³⁴", fontsize=10)
    ax_full.grid(True, which="both", axis="y", color="#e1e0d9", linewidth=0.6)
    ax_full.set_axisbelow(True)

    fig.suptitle("Same gradient, five step sizes — gradual descent vs. overshoot", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved → {save_path}")
    plt.close(fig)


def generate_visuals(X, y, w_star):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_loss_landscape(X, y, w_star, save_path=f"{OUTPUT_DIR}/loss_landscape.png")
    plot_learning_rate_behavior(X, y, save_path=f"{OUTPUT_DIR}/learning_rate_behavior.png")


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
    w_star, *_ = section_gd_variants(X, y)
    section_learning_rate(X, y)

    print("\n" + "=" * 70)
    print("Generating visuals …")
    print("=" * 70)
    generate_visuals(X, y, w_star)

    print("\nDone.")


if __name__ == "__main__":
    main()
