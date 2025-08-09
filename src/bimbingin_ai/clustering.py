from __future__ import annotations
from typing import Optional, Iterable, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def pick_k_by_silhouette(X6, ks: Iterable[int]) -> Tuple[int, Dict[str, float]]:
    best_k, best_s, best_db = None, -1.0, None
    for k in ks:
        if k <= 1 or k > len(X6):
            continue
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(X6)
        if len(set(labels)) == 1:
            continue
        s = silhouette_score(X6, labels)
        db = davies_bouldin_score(X6, labels)
        if s > best_s or (abs(s - best_s) < 1e-6 and (best_db is None or db < best_db)):
            best_k, best_s, best_db = k, s, db
    return best_k or 2, {"silhouette": float(best_s), "davies_bouldin": float(best_db)}

def run_kmeans(X6, k: Optional[int] = None, mode="auto", ks_range=range(4,9), seed=42):
    if mode == "auto":
        k_chosen, metrics = pick_k_by_silhouette(X6, ks_range)
    else:
        assert k is not None and 2 <= k <= len(X6), "invalid k"
        k_chosen = k
        tmp = KMeans(n_clusters=k_chosen, n_init="auto", random_state=seed).fit(X6)
        labels = tmp.labels_
        if len(set(labels)) > 1:
            s = silhouette_score(X6, labels)
            db = davies_bouldin_score(X6, labels)
        else:
            s, db = -1.0, 1e9
        metrics = {"silhouette": float(s), "davies_bouldin": float(db)}
    model = KMeans(n_clusters=k_chosen, n_init="auto", random_state=seed).fit(X6)
    return model.labels_, model.cluster_centers_, metrics, k_chosen
