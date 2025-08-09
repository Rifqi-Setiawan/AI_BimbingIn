from flask import Flask, request, jsonify
import numpy as np
# CATEGORIES tidak dipakai—boleh dihapus dari import
from .preprocess import preprocess_25_and_6  # , CATEGORIES
from .clustering import run_kmeans
from .balancer import balance_with_ilp

# ⬇️ Tambahkan ini
from sklearn.metrics import silhouette_score

app = Flask(__name__)

DIM6 = ["avoidant","collaborative","competitive","dependent","independent","participative"]

def cluster_quality_score(X, labels):
    try:
        if len(set(labels)) >= 2 and all((labels == c).sum() > 1 for c in set(labels)):
            sil = silhouette_score(X, labels)
            return round(((sil + 1) / 2) * 100.0, 1)  # 0..100%
    except Exception:
        pass

    # fallback ke R²
    global_mean = X.mean(axis=0)
    tss = float(((X - global_mean) ** 2).sum())
    wss = sum(((X[labels == c] - X[labels == c].mean(axis=0)) ** 2).sum()
              for c in set(labels))
    r2 = 0.0 if tss <= 1e-12 else 1.0 - (wss / tss)
    return round(max(0.0, min(1.0, r2)) * 100.0, 1)

def to_percent_readable(scores_readable: dict[str, float], ndigits: int = 1):
    total = sum(scores_readable.get(d, 0.0) for d in DIM6)
    if total <= 0:
        return {d: 0.0 for d in DIM6}
    perc = {d: (scores_readable[d] / total) * 100.0 for d in DIM6}
    rounded = {d: round(perc[d], ndigits) for d in DIM6}
    diff = round(100.0 - sum(rounded.values()), ndigits)
    if diff != 0:
        top = max(rounded, key=rounded.get)
        rounded[top] = round(rounded[top] + diff, ndigits)
    return rounded

def dominant_and_secondary(perc: dict[str, float]):
    order = sorted(DIM6, key=lambda d: perc[d], reverse=True)
    return order[0], order[1]

def summarize_groups(labels, student_ids, centroids6_norm, X6_raw):
    groups = []
    k = centroids6_norm.shape[0]
    for c in range(k):
        members_idx = np.where(labels == c)[0]
        members_ids = [student_ids[i] for i in members_idx]
        raw_block = X6_raw[members_idx] if len(members_idx) else np.zeros((0,6))
        mean_scores = raw_block.mean(axis=0) if len(raw_block) else np.zeros(6)

        scores_readable = {cat: float(val) for cat, val in zip(DIM6, mean_scores)}
        percentages = to_percent_readable(scores_readable, ndigits=1)
        dom, sec = dominant_and_secondary(percentages)

        groups.append({
            "group_id": c+1,
            "label": f"Dominan: {dom}; Sekunder: {sec}",
            "size": len(members_ids),
            "student_ids": members_ids,
            "percentages": percentages
        })
    return groups

@app.post("/cluster")
def cluster():
    req = request.get_json(force=True) or {}
    mode = req.get("mode", "auto")
    k = req.get("k", None)
    constraints = req.get("constraints") or {}
    min_group_size = int(constraints.get("min_group_size", 20))
    max_group_size = int(constraints.get("max_group_size", 36))
    balance = bool(constraints.get("balance", True))
    seed_used = int(constraints.get("seed", 42))
    lambda_move_penalty = float(constraints.get("lambda_move_penalty", 0.0))

    students = req.get("students", [])
    if not students:
        return jsonify(error="students cannot be empty"), 400

    student_ids = [p["id"] for p in students]
    X25 = np.array([p["answers"] for p in students], dtype=float)
    if X25.shape[1] != 25:
        return jsonify(error="Each student must have 25 answers"), 400

    pp = preprocess_25_and_6(X25, do_mean_center=True, do_l2_on_6d=True)
    X6_norm, X6_raw = pp["X6_norm"], pp["X6_raw"]

    labels, centroids6, metrics, k_chosen = run_kmeans(
        X6_norm, k=k, mode=mode, ks_range=range(4,9), seed=seed_used
    )

    if balance:
        k_now = centroids6.shape[0]
        if min_group_size * k_now > len(student_ids) or max_group_size * k_now < len(student_ids):
            return jsonify(error="min/max group size infeasible"), 400
        labels, _ = balance_with_ilp(
            X6_norm, labels, centroids6,
            min_group_size, max_group_size, lambda_move_penalty
        )
        centroids6 = np.vstack([
            X6_norm[labels == c].mean(axis=0) if np.any(labels == c) else centroids6[c]
            for c in range(k_now)
        ])

    groups = summarize_groups(labels, student_ids, centroids6, X6_raw)

    quality_cluster_percent = cluster_quality_score(X6_norm, labels)

    return jsonify({
        "k_chosen": k_chosen,
        "seed_used": seed_used,
        "quality_cluster_percent": quality_cluster_percent,
        "groups": groups
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)
