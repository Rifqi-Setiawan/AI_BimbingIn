from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np

QUESTION_TO_CATEGORY: Dict[str, List[int]] = {
    "collaborative": [0, 6, 12, 18, 24],
    "independent":   [1, 7, 13, 19],
    "dependent":     [2, 8, 14, 20],
    "competitive":   [3, 9, 15, 21],
    "participative": [4, 10, 16, 22],
    "avoidant":      [5, 11, 17, 23],
}
CATEGORIES = list(QUESTION_TO_CATEGORY.keys())

def imputate_median(X25):
    X = X25.copy()
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    return X

def mean_center_per_student(X25):
    return X25 - X25.mean(axis=1, keepdims=True)

def clip_scale_1_5(X25):
    return np.clip(X25, 1.0, 5.0)

def map_25_to_6(X25):
    return np.column_stack([X25[:, idxs].mean(axis=1) for idxs in QUESTION_TO_CATEGORY.values()])

def zscore(X, eps=1e-8):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sd, mu, sd

def l2_normalize(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def preprocess_25_and_6(X25, do_mean_center=True, do_l2_on_6d=True):
    X25p = imputate_median(X25)
    if do_mean_center:
        X25p = mean_center_per_student(X25p)
    X25p = clip_scale_1_5(X25p)
    X6_raw = map_25_to_6(X25p)
    X6z, mu, sd = zscore(X6_raw)
    X6_norm = l2_normalize(X6z) if do_l2_on_6d else X6z
    return {"X25": X25p, "X6_norm": X6_norm, "X6_raw": X6_raw, "mu6": mu, "sd6": sd}
