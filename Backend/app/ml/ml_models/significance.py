"""
uncertainty tools for AUC on autocorrelated (Overlapping-label) time series.''
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def blocking_bootstrap_auc_ci(
        y_true  , scores , block : int , n_boot : int = 2000, alpha : float = 0.05 , seed : int = 42
):

    """
    removing block bootstrap CI for ROC AUC returns (auc , lo , hi)
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n = len(y_true)
    block = max(1 , min(block , n ))
    n_block = int(np.ceil(n/block))
    rng = np.random.default_rng(seed)


    aucs = []

    for _ in range(n_boot):
        starts = rng.integers( 0 , n - block + 1 , size = n_block)
        idx = np.concatenate([np.arange(s , s + block) for s in starts ])[:n]
        yb = y_true[idx]

        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, scores[idx]))


    lo , hi = np.percentile(aucs , [ 100 * alpha / 2 , 100 * (1 - alpha/2)])
    return float(roc_auc_score(y_true , scores))  , float(lo) , float(hi)


def circular_shift_pvalue(y_true, scores ,  n_perm : int = 2000 , seed : int = 42):
    """
    two sided p valye for H0 : scores carry no information about ytrue ,  null = auc after circularly shifting scores against labels , which keeps the autocorrelation of both series intact ,

    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    min_shift = max(1 , n // 20)

    observed = abs(roc_auc_score(y_true , scores ) - 0.5 )
    shifts = rng.integers(min_shift , n - min_shift , size = n_perm)
    null = np.array([abs(roc_auc_score(y_true, np.roll(scores, s)) - 0.5) for s in shifts])
    return float((1 + np.sum(null >= observed)) / (n_perm + 1))


def best_direction_auc(x , y_true):
    """
    auc of a single raw feature used as a score , in whichever direction is better  , deliberatly generous so it is hard baseline to beat
    """

    a = roc_auc_score(y_true ,  x)
    return float(max(a , 1 - a))






