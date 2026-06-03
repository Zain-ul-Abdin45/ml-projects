"""
What "Learning from Data" Actually Means — Before Any Model
Act 1, Article 1 of the ML Series

Companion code for: https://github.com/Zain-ul-Abdin45/APIs_Development

Standalone script. Downloads NYC TLC January 2024 data, cleans it with
intent, engineers features, and produces a ready-to-model train/test split.
No model.fit() — that is the point.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Pinned data URLs ──────────────────────────────────────────────────────────
DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi+_zone_lookup.csv"


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Pull January 2024 yellow cab data directly from TLC. ~250 MB parquet."""
    print("Loading TLC data …")
    df = pd.read_parquet(DATA_URL)
    print(f"Shape : {df.shape}")
    print(df.dtypes)
    return df


# ── Step 2: Understand the raw numbers ───────────────────────────────────────

def inspect_raw(df: pd.DataFrame) -> None:
    print("\n── Raw describe ──")
    cols = ["passenger_count", "trip_distance", "fare_amount",
            "tip_amount", "total_amount"]
    print(df[cols].describe().to_string())

    print("\n── Missing values ──")
    missing = df.isnull().sum() / len(df) * 100
    print(missing[missing > 0].sort_values(ascending=False))


# ── Step 3: Clean with intent, not habit ─────────────────────────────────────

def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every filter here is a decision with a domain reason.
    Not quantile-based cutoffs — explicit business rules.
    """
    # Positive fare — negative fares are correction entries in the billing system
    df = df[df["fare_amount"] > 0]

    # Tip cannot be negative
    df = df[df["tip_amount"] >= 0]

    # At least one passenger
    df = df[df["passenger_count"] >= 1]

    # NYC yellow cabs fit 4 passengers; 5 is the hard limit with the front seat
    df = df[df["passenger_count"] <= 5]

    # A trip requires some distance
    df = df[df["trip_distance"] > 0]

    # Cross-borough is rarely over 50 miles
    df = df[df["trip_distance"] <= 50]

    # NYC base fare is $3.00 in 2024; anything below is a system entry
    df = df[df["fare_amount"] >= 3.0]
    df = df[df["fare_amount"] <= 200]

    # Derive duration before filtering on it
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[df["trip_duration_min"] > 1]
    df = df[df["trip_duration_min"] <= 120]

    # Only standard rate (1) and JFK flat rate (2) — negotiated fares skew the fare structure
    df = df[df["RatecodeID"].isin([1, 2])]

    return df.reset_index(drop=True)


# ── Step 4: Feature engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw columns are containers for features — a timestamp is not a feature,
    but the hour, day, and whether it is rush hour are.
    """
    dt = df["tpep_pickup_datetime"]

    # Time features
    df["pickup_hour"]      = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek   # 0 = Monday
    df["pickup_month"]     = dt.dt.month
    df["is_weekend"]       = (dt.dt.dayofweek >= 5).astype(int)
    df["is_rush_hour"]     = (
        ((dt.dt.hour >= 7)  & (dt.dt.hour <= 9)) |
        ((dt.dt.hour >= 16) & (dt.dt.hour <= 19))
    ).astype(int)
    df["is_late_night"]    = (
        (dt.dt.hour >= 22) | (dt.dt.hour <= 5)
    ).astype(int)

    # Speed — a proxy for traffic conditions
    # trip_distance in miles, duration in minutes
    df["avg_speed_mph"] = (
        df["trip_distance"] / (df["trip_duration_min"] / 60)
    ).clip(upper=80)  # 25 mph speed limit; 80 caps GPS/logging errors

    # Normalised cost signal
    df["fare_per_mile"] = (
        df["fare_amount"] / df["trip_distance"]
    ).clip(upper=50)

    # Cash tips are not logged by the meter system — this is a structural data gap
    df["is_card_payment"] = (df["payment_type"] == 1).astype(int)

    # JFK flat-rate flag
    df["is_jfk_rate"] = (df["RatecodeID"] == 2).astype(int)

    return df


# ── Step 5: Cash tip trap — the most important check ─────────────────────────

def check_cash_tip_gap(df: pd.DataFrame) -> None:
    """
    Cash tips are handed to the driver and never logged. Training on both
    payment types teaches the model that cash trips tip nothing — which is wrong.
    """
    cash = df[df["payment_type"] == 2]
    card = df[df["payment_type"] == 1]

    print(f"\nCash trips with tip_amount > 0 : {(cash['tip_amount'] > 0).mean():.2%}")
    print(f"Card trips with tip_amount > 0 : {(card['tip_amount'] > 0).mean():.2%}")
    print("\nFilter to card payments only before modelling.")


# ── Step 6: Distribution analysis ────────────────────────────────────────────

def plot_tip_distributions(df_model: pd.DataFrame,
                           save_path: str = "tip_distributions.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_model["tip_amount"], bins=100,
                 color="#2c3e50", edgecolor="none")
    axes[0].set_title("Tip Amount — Raw")
    axes[0].set_xlabel("Tip ($)")
    axes[0].set_ylabel("Count")

    df_model = df_model.copy()
    df_model["tip_pct"] = df_model["tip_amount"] / df_model["fare_amount"] * 100
    axes[1].hist(df_model["tip_pct"].clip(0, 50), bins=100,
                 color="#2c3e50", edgecolor="none")
    axes[1].set_title("Tip as % of Fare")
    axes[1].set_xlabel("Tip (%)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")
    plt.show()


def check_20pct_spike(df_model: pd.DataFrame) -> None:
    df_model = df_model.copy()
    df_model["tip_pct"] = df_model["tip_amount"] / df_model["fare_amount"] * 100
    at_20 = ((df_model["tip_pct"] >= 19.5) & (df_model["tip_pct"] <= 20.5)).sum()
    print(f"\nTrips tipping exactly ~20%: {at_20:,}  ({at_20 / len(df_model):.1%})")
    print("This is a button on the payment terminal, not a tipping decision.")


# ── Step 7: Correlation analysis ─────────────────────────────────────────────

def correlation_analysis(df_model: pd.DataFrame) -> None:
    df_model = df_model.copy()
    df_model["tip_pct"] = df_model["tip_amount"] / df_model["fare_amount"] * 100

    features = [
        "trip_distance", "fare_amount", "pickup_hour",
        "is_weekend", "is_rush_hour", "is_late_night",
        "avg_speed_mph", "fare_per_mile", "trip_duration_min",
        "passenger_count", "is_jfk_rate",
    ]

    print("\n── Correlation with tip_amount ──")
    corr_amt = df_model[features + ["tip_amount"]].corr()["tip_amount"].drop("tip_amount")
    print(corr_amt.sort_values(ascending=False).to_string())

    print("\n── Correlation with tip_pct ──")
    corr_pct = df_model[features + ["tip_pct"]].corr()["tip_pct"].drop("tip_pct")
    print(corr_pct.sort_values(ascending=False).to_string())

    print("\nTarget change inverts the correlation structure.")
    print("The question determines the signal — not just the algorithm.")


# ── Step 8: Add zone features and build final dataset ─────────────────────────

def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    zones    = pd.read_csv(ZONE_URL)
    zone_map = zones.set_index("LocationID")["Borough"].to_dict()

    df["pickup_borough"]  = df["PULocationID"].map(zone_map)
    df["dropoff_borough"] = df["DOLocationID"].map(zone_map)

    borough_enc = {b: i for i, b in enumerate(df["pickup_borough"].dropna().unique())}
    df["pickup_borough_enc"]  = df["pickup_borough"].map(borough_enc)
    df["dropoff_borough_enc"] = df["dropoff_borough"].map(borough_enc)
    return df


MODEL_FEATURES = [
    "trip_distance",
    "trip_duration_min",
    "fare_amount",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "avg_speed_mph",
    "fare_per_mile",
    "passenger_count",
    "is_jfk_rate",
    "pickup_borough_enc",
    "dropoff_borough_enc",
]
TARGET = "tip_pct"


# ── Step 9: Time-based split ──────────────────────────────────────────────────

def build_train_test_split(df_model: pd.DataFrame):
    """
    Random split leaks time — the model would be evaluated on the past.
    In production it only ever sees the future. Use a time boundary.
    """
    df_model = df_model.copy()
    df_model[TARGET] = (df_model["tip_amount"] / df_model["fare_amount"] * 100).clip(0, 50)

    df_final = df_model[MODEL_FEATURES + [TARGET, "tpep_pickup_datetime"]].dropna()
    df_final = df_final.sort_values("tpep_pickup_datetime")

    print(f"\nFinal dataset shape : {df_final[MODEL_FEATURES + [TARGET]].shape}")
    print(f"\nTarget variable summary:")
    print(df_final[TARGET].describe().to_string())

    # Last 25% of January as test set
    cutoff     = df_final["tpep_pickup_datetime"].quantile(0.75)
    train_mask = df_final["tpep_pickup_datetime"] <= cutoff

    X_train = df_final.loc[ train_mask, MODEL_FEATURES]
    X_test  = df_final.loc[~train_mask, MODEL_FEATURES]
    y_train = df_final.loc[ train_mask, TARGET]
    y_test  = df_final.loc[~train_mask, TARGET]

    print(f"\nTraining samples : {len(X_train):,}")
    print(f"Test samples     : {len(X_test):,}")

    return X_train, y_train, X_test, y_test


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("What 'Learning from Data' Actually Means — Before Any Model")
    print("=" * 55)

    df = load_raw()

    print("\n[Step 1] Raw data inspection")
    inspect_raw(df)

    print("\n[Step 2] Cleaning")
    df_clean = clean_trips(df)
    print(f"Original rows : {len(df):,}")
    print(f"After cleaning: {len(df_clean):,}")
    print(f"Removed       : {len(df) - len(df_clean):,}  "
          f"({(1 - len(df_clean)/len(df))*100:.1f}%)")

    print("\n[Step 3] Feature engineering")
    df_feat = engineer_features(df_clean)

    print("\n[Step 4] Cash tip gap check")
    check_cash_tip_gap(df_feat)

    # Card payments only from here forward
    df_model = df_feat[df_feat["is_card_payment"] == 1].copy()
    print(f"\nCard payment trips: {len(df_model):,}")

    print("\n[Step 5] Distribution analysis")
    plot_tip_distributions(df_model)
    check_20pct_spike(df_model)

    print("\n[Step 6] Correlation analysis")
    correlation_analysis(df_model)

    print("\n[Step 7] Zone features + final split")
    df_model = add_zone_features(df_model)
    X_train, y_train, X_test, y_test = build_train_test_split(df_model)

    print("\nDone. X_train / y_train / X_test / y_test are ready.")
    print("No model.fit() — that is the next article.")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
