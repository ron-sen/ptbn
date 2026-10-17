
import pandas as pd
import numpy as np

from app.ml.ml_models.labeling import compute_forward_return_label
from app.ml.ml_models.train_lr import train_lr_walk_forward


# genrate fake market data 

np.random.seed(42)

N = 100

timestamps = pd.date_range(
    start="2026-01-01 09:15:00",
    periods=N,
    freq="1min",
)

# Start around 100 and create a somewhat random walk

close = pd.Series(
    100 + np.cumsum(np.random.normal(0, 0.3, N))
)

# Fake ATR values

atr = pd.Series(
    np.random.uniform(0.2, 0.6, N)
)

# Fake RSI

rsi = pd.Series(
    np.random.uniform(25, 75, N)
)

# Fake PPO

ppo = pd.Series(
    np.random.uniform(-2, 2, N)
)

# test labeling function
labels = compute_forward_return_label(
    close=close,
    atr=atr,
    n_candles_ahead=10,
    atr_multiplier=1.0,
)


# build dataframe expected by LR
df = pd.DataFrame({
    "timestamp": timestamps,
    "rsi": rsi,
    "ppo": ppo,
    "atr": atr,
    "label": labels,
})


# Remove rows where label is NaN
df = df.dropna(subset=["label"]).reset_index(drop=True)

# Label must be integer for training
df["label"] = df["label"].astype(int)


print("\n================ DATASET ================\n")
print(df.head(15))

print("\nDataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())


# sanity check the label manually

print("\n================ LABEL CHECK ================\n")

for i in range(min(10, len(df))):
    print(
        f"Row {i:02d} | "
        f"Close={close.iloc[i]:.2f} | "
        f"ATR={atr.iloc[i]:.2f} | "
        f"Threshold={close.iloc[i] + atr.iloc[i]:.2f} | "
        f"Label={labels.iloc[i]}"
    )


# train lr using walk forward cv

print("\n================ TRAINING ================\n")

results = train_lr_walk_forward(df)




# inspect every fold 

for result in results:

    print(f"\n----------- Fold {result['fold']} -----------")

    print("Test indices:")
    print(result["test_index"].tolist())

    print("\nTrue labels:")
    print(result["y_true"])

    print("\nPredicted probabilities:")
    print(
        np.round(result["y_pred_proba"], 3)
    )

    print("\nModel coefficients:")
    print(result["model"].coef_)

    print("\nModel intercept:")
    print(result["model"].intercept_)


print("\n================ DONE ================\n")