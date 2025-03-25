import numpy as np
import cvxpy as cp
import networkx as nx

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from graphs import laplacian_matrix, compute_eigenvalues


def certify_conformal_rigidity(G, use_prop3_2=False, tol=1e-8, eps=1e-11):
    m = G.number_of_edges()
    L = laplacian_matrix(G)
    eigvals, _ = compute_eigenvalues(L)
    lambda_n = eigvals[-1]  # largest eigenvalue
    lambda_2 = eigvals[1]  # second smallest eigenvalue; lambda1 = 0

    a, b, c, details_2 = certify_conformal_rigidity_lambda_2(
        G, L, m, lambda_2, tol=tol, eps=eps
    )

    if not (a if use_prop3_2 else b and c):
        return False, (details_2, None)

    a, b, c, details_n = certify_conformal_rigidity_lambda_n(
        G, L, m, lambda_n, tol=tol, eps=eps
    )

    return a if use_prop3_2 else b and c, (details_2, details_n)


def certify_conformal_rigidity_lambda_2(G, L, m, lambda_2, tol=1e-8, eps=1e-11):
    """
    Check the lambda_2 certificate conditions (from Proposition 3.3) for conformal rigidity.
    That is, for the dual certificate X we want:
      (i) Trace(X) ≈ |E| / lambda_2,
      (ii) For every edge (i,j), X_ii + X_jj - 2X_ij ≈ 1,
      (iii) X is approximately an eigenmatrix for L with eigenvalue lambda_2 (i.e. L X ≈ lambda_2 X).

    Returns:
       i: whether i holds,
       ii: whether ii holds,
       iii: whether iii holds,
       details: dictionary with computed values.
    """
    target_trace = m / lambda_2

    # Solve the dual SDP for lambda2.
    X_opt, opt_val = solve_dual_sdp_lambda_2(G, eps=eps)

    details = {
        "lambda_2": lambda_2,
        "num_edges": m,
        "target_trace": target_trace,
        "sdp_trace": opt_val,
        "max_edge_violation": None,
        "eigenvector_residual": None,
    }

    # Check if the optimal trace is (approximately) equal to m/lambda2.
    trace_diff = np.abs(opt_val - target_trace)

    # Now check the edge constraints: for every edge (i,j), we want X_ii+X_jj-2X_ij == 1.
    max_edge_violation = 0
    for i, j in G.edges():
        diff_ij = np.abs(X_opt[i, i] + X_opt[j, j] - 2 * X_opt[i, j] - 1)  # type: ignore
        if diff_ij > max_edge_violation:
            max_edge_violation = diff_ij
    details["max_edge_violation"] = max_edge_violation

    # Finally, check that X_opt is (approximately) an eigenmatrix for L with eigenvalue lambda2.
    # Compute the residual norm of L X_opt - lambda2 X_opt.
    LX = L @ X_opt
    residual = LX - lambda_2 * X_opt
    eigenvector_residual = np.linalg.norm(residual, ord="fro")
    details["eigenvector_residual"] = eigenvector_residual

    return (
        trace_diff < tol,
        max_edge_violation < tol,
        eigenvector_residual < tol,
        details,
    )


def certify_conformal_rigidity_lambda_n(G, L, m, lambda_n, tol=1e-3, eps=1e-11):
    """
    Check the lambda_n certificate conditions (from Proposition 3.3) for conformal rigidity.
    That is, for the dual certificate Y we want:
      (i) Trace(Y) ≈ |E| / lambda_n,
      (ii) For every edge (i,j), Y_ii + Y_jj - 2Y_ij ≈ 1,
      (iii) Y is approximately an eigenmatrix for L with eigenvalue lambda_n (i.e. L Y ≈ lambda_n Y).

    Returns:
       i: whether i holds,
       ii: whether ii holds,
       iii: whether iii holds,
       details: dictionary with computed values.
    """
    target_trace = m / lambda_n  # The optimal trace should be |E|/lambda_n

    # Solve the dual SDP for lambda_n
    Y_opt, opt_val = solve_dual_sdp_lambda_n(G, eps=eps)

    details = {
        "lambda_n": lambda_n,
        "num_edges": m,
        "target_trace": target_trace,
        "sdp_trace": opt_val,
        "max_edge_violation": None,
        "eigenmatrix_residual": None,
    }

    # Check the trace condition.
    trace_diff = np.abs(opt_val - target_trace)

    # Check the edge constraints: for each edge (i,j) we want Y_ii + Y_jj - 2Y_ij ≈ 1.
    max_edge_violation = 0
    for i, j in G.edges():
        violation = np.abs(Y_opt[i, i] + Y_opt[j, j] - 2 * Y_opt[i, j] - 1)  # type: ignore
        if violation > max_edge_violation:
            max_edge_violation = violation
    details["max_edge_violation"] = max_edge_violation

    # Check that Y_opt is approximately an eigenmatrix for L with eigenvalue lambda_n,
    # i.e., compute the Frobenius norm of (L Y_opt - lambda_n * Y_opt).
    residual = L @ Y_opt - lambda_n * Y_opt
    eigenmatrix_residual = np.linalg.norm(residual, ord="fro")
    details["eigenmatrix_residual"] = eigenmatrix_residual

    return (
        trace_diff < tol,
        max_edge_violation < tol,
        eigenmatrix_residual < tol,
        details,
    )


def solve_dual_sdp_lambda_2(G, eps=1e-11):
    """
    Solve the dual SDP for lambda_2:
        maximize Trace(X)
        subject to X_ii + X_jj - 2*X_ij <= 1 for each edge (i,j)
                   1^T X 1 = 0, X psd.
    Returns:
        X_opt: optimal X matrix (as a NumPy array)
        opt_val: optimal value (trace(X_opt))
    """
    n = G.number_of_nodes()
    # Define the variable: symmetric n x n matrix X.
    X = cp.Variable((n, n), symmetric=True)

    constraints = []
    # Constraint: For every edge (i,j) in G, enforce X_ii + X_jj - 2 X_ij <= 1.
    for i, j in G.edges():
        constraints.append(X[i, i] + X[j, j] - 2 * X[i, j] <= 1)
    # Constraint: X is orthogonal to the constant vector.
    ones = np.ones((n, 1))
    constraints.append(
        cp.sum(cp.multiply(ones, X @ ones)) == 0
    )  # equivalent to 1^T X 1 = 0
    # PSD constraint.
    constraints.append(X >> 0)

    # Objective: maximize Trace(X)
    objective = cp.Maximize(cp.trace(X))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, eps=eps)
    # prob.solve(solver=cp.CLARABEL, max_iter=1000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: SDP did not converge to an optimal solution.")

    X_opt = X.value
    opt_val = np.trace(X_opt)  # type: ignore
    return X_opt, opt_val


def solve_dual_sdp_lambda_n(G, eps=1e-11):
    """
    Solve the dual SDP for lambda_n:
        minimize Trace(Y)
        subject to Y_ii + Y_jj - 2*Y_ij >= 1 for every edge (i,j)
                   1^T Y 1 = 0,
                   Y is PSD.
    Returns:
        Y_opt: optimal Y matrix (as a NumPy array)
        opt_val: optimal value (trace(Y_opt))
    """
    n = G.number_of_nodes()
    Y = cp.Variable((n, n), symmetric=True)

    constraints = []
    # For each edge (i,j), add: Y_ii + Y_jj - 2*Y_ij >= 1.
    for i, j in G.edges():
        constraints.append(Y[i, i] + Y[j, j] - 2 * Y[i, j] >= 1)

    # Enforce orthogonality to the constant vector: 1^T Y 1 = 0.
    ones = np.ones((n, 1))
    constraints.append(cp.sum(cp.multiply(ones, Y @ ones)) == 0)

    # Y must be positive semidefinite.
    constraints.append(Y >> 0)

    # Objective: minimize Trace(Y)
    objective = cp.Minimize(cp.trace(Y))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, eps=eps)
    # prob.solve(solver=cp.CLARABEL, max_iter=1000)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("Warning: SDP did not converge to an optimal solution.")

    Y_opt = Y.value
    opt_val = np.trace(Y_opt)  # type: ignore
    return Y_opt, opt_val
