import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from bimbingin_ai.preprocess import preprocess_25_and_6
from bimbingin_ai.clustering import run_kmeans
from bimbingin_ai.balancer import balance_with_ilp
import numpy as np

def main():
    rng = np.random.default_rng(42)
    X25 = np.clip(rng.normal(3.5, 0.7, size=(100,25)), 1, 5)
    pp = preprocess_25_and_6(X25)
    labels, centroids6, metrics, k = run_kmeans(pp["X6_norm"], mode="auto", ks_range=range(4,9), seed=42)
    print("Auto-k:", k, metrics)
    labels2, diag = balance_with_ilp(pp["X6_norm"], labels, centroids6, 20, 30, 0.05)
    print("Balanced sizes:", diag["sizes"])

if __name__ == "__main__":
    main()
