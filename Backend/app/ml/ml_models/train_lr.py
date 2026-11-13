
"""
trains logisitic regression using walk forward (timesseriesplit) validation - never tests on data chronologically before what it trained on.

"""

from typing import Any
import pandas as pd
import numpy as np  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler #  after improvision
from app.ml.ml_models.dataset import N_CANDLES_AHEAD

FEATURE_COLUMNS = [ "rsi" , "ppo" , "atr_pct"]
N_SPLITS = 5 

def train_lr_walk_forward( df : pd.DataFrame , gap : int = N_CANDLES_AHEAD , class_weight = "balanced") -> list[dict[str , Any]] :

    """
    df must be sorted chronologically , containing FEATURE_COLUMNS + 'label'. 

    returns one result dict per fold : trained model , test - row indices, true labels and predicted probabilities - ready for your evaluation step (precision / recall / confusion matrix , comparison against fuzzy scores )

    """

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS , gap=gap)
    fold_results = []

    for fold_index , (train_idx , test_idx) in enumerate(tscv.split(X)):
        X_train , X_test = X[train_idx] , X[test_idx]
        y_train , y_test = y[train_idx] , y[test_idx]

        # scale  feature per fold to ensure zero mean and unit variance 
        # after improvision

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        model = LogisticRegression(max_iter= 1000 , class_weight= class_weight)
        model.fit(X_train_scaled , y_train)

        predicted_probabilities = model.predict_proba(X_test_scaled)[ : , 1]

        fold_results.append({
            "fold": fold_index,
            "model": model,
            "scaler" : scaler , 
            "test_index": df.index[test_idx],
            "y_true": y_test,
            "y_pred_proba": predicted_probabilities,
            "train_majority_label": int(np.round(y_train.mean())),
        })

    return fold_results