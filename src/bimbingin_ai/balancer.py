from ortools.linear_solver import pywraplp
import numpy as np

def squared_euclidean(a, b):
    d = a - b
    return float(np.dot(d, d))

def build_cost_matrix(X6, centroids6):
    n, k = X6.shape[0], centroids6.shape[0]
    return np.array([[squared_euclidean(X6[i], centroids6[c]) for c in range(k)] for i in range(n)], dtype=float)

def balance_with_ilp(X6, labels, centroids6, min_group_size, max_group_size, lambda_move_penalty=0.0):
    n, k = X6.shape[0], centroids6.shape[0]

    # Cek feasibility batas ukuran grup lebih dulu
    if min_group_size * k > n or max_group_size * k < n:
        raise ValueError(
            f"Constraint ukuran grup tidak feasible: n={n}, k={k}, "
            f"min={min_group_size}, max={max_group_size}"
        )

    C = build_cost_matrix(X6, centroids6)
    # Penalti jika pindah dari label awal
    for i in range(n):
        for c in range(k):
            if c != labels[i]:
                C[i, c] += float(lambda_move_penalty)

    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("Solver CBC tidak tersedia. Pastikan ortools terpasang dengan backend CBC.")

    # Variabel biner x[i,c] = 1 jika siswa i ditugaskan ke cluster c
    x = {(i, c): solver.BoolVar(f"x_{i}_{c}") for i in range(n) for c in range(k)}

    # Setiap siswa tepat di satu cluster
    for i in range(n):
        solver.Add(sum(x[i, c] for c in range(k)) == 1)

    # Batas ukuran tiap cluster
    for c in range(k):
        solver.Add(sum(x[i, c] for i in range(n)) >= int(min_group_size))
        solver.Add(sum(x[i, c] for i in range(n)) <= int(max_group_size))

    # Objektif: minimalkan biaya penugasan
    objective = solver.Objective()
    for i in range(n):
        for c in range(k):
            objective.SetCoefficient(x[i, c], float(C[i, c]))
    objective.SetMinimization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("ILP tidak menemukan solusi feasible.")

    # Ambil label baru: untuk tiap i, cari c dengan nilai 1
    new_labels = np.array([
        max(range(k), key=lambda c: x[i, c].solution_value())
        for i in range(n)
    ], dtype=int)

    sizes = [int((new_labels == c).sum()) for c in range(k)]
    total_cost = float(sum(C[i, new_labels[i]] for i in range(n)))
    return new_labels, {"total_cost": total_cost, "sizes": sizes}
