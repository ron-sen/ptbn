
"""
train random forest using the same walk forward ( timeseries-split) validation as train_lr.py , same fature , same label , same fold structure  , so results are directly comparable to LR fold by fold

"""

from typing import Any
import numpy as np 
import  pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from app.ml.ml_models.dataset import N_CANDLES_AHEAD

FEATURE_COLUMNS =  [ "rsi" , "ppo" , "atr_pct"]
N_SPLITS = 5 
N_ESTIMATORS = 200 
MAX_DEPTH = 5 # kept it  shallow deliberately 

def train_rf_walk_forward(df: pd.DataFrame , gap : int = N_CANDLES_AHEAD) -> list[dict[str , Any]]:

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits= N_SPLITS , gap=gap)
    fold_results = []

    for fold_index , ( train_idx , test_idx) in enumerate(tscv.split(X)):

        X_train , X_test = X[train_idx] , X[test_idx]
        y_train , y_test = y[train_idx] , y[test_idx]

        model = RandomForestClassifier(
            n_estimators = N_ESTIMATORS , 
            max_depth = MAX_DEPTH , 
            class_weight= "balanced" ,
            random_state = 42 ,
        )
        model.fit(X_train , y_train)

        predicted_probabilites = model.predict_proba(X_test)[: , 1 ]

        fold_results.append({
            "fold" : fold_index ,
            "model" : model , 
            "test_index" : df.index[test_idx],
            "y_true" : y_test , 
            "y_pred_proba" : predicted_probabilites , 
            "feature_importances" : dict(zip(FEATURE_COLUMNS , model.feature_importances_)) ,
            "train_majority_label": int(np.round(y_train.mean())),
        })

    return fold_results