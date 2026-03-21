import math
import numpy as np
from .model import MertonJumpDiffusion2D


def compute_greens_weights(
    model: MertonJumpDiffusion2D,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dtau: float,
    tol: float = 1e-12,
) -> np.ndarray:
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    C = dtau * model.C_tilde
    beta = dtau * model.beta_tilde
    theta = -(model.r + model.lam) * dtau
    C_M = model.C_M
    mu_tilde = model.mu_tilde_vec
    lam_dt = model.lam * dtau

    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    z = np.stack([X, Y], axis=-1)

    C_inv_0 = np.linalg.inv(C)
    det_C_0 = np.linalg.det(C)

    diff_0 = beta[np.newaxis, np.newaxis, :] + z
    quad_0 = np.einsum('...i,ij,...j', diff_0, C_inv_0, diff_0)

    weights = (dx * dy / (2.0 * np.pi * np.sqrt(det_C_0))) * np.exp(theta - 0.5 * quad_0)

    k = 0
    while True:
        k += 1

        C_k = C + k * C_M
        C_k_inv = np.linalg.inv(C_k)
        det_C_k = np.linalg.det(C_k)

        diff_k = beta[np.newaxis, np.newaxis, :] + z + k * mu_tilde[np.newaxis, np.newaxis, :]
        quad_k = np.einsum('...i,ij,...j', diff_k, C_k_inv, diff_k)

        poisson_weight = lam_dt**k / math.factorial(k)
        term_k = poisson_weight * (dx * dy / (2.0 * np.pi * np.sqrt(det_C_k))) * np.exp(theta - 0.5 * quad_k)

        weights += term_k

        test = (np.exp(-(model.r + model.lam) * dtau)
                / (2.0 * np.pi * np.sqrt(np.linalg.det(C)))
                * (np.e * lam_dt)**(k + 1) / (k + 1)**(k + 1))

        if test < tol:
            break

        if k > 500:
            print(f"Warning: Green's function series did not converge after {k} terms")
            break

    return weights


def truncation_bound(model: MertonJumpDiffusion2D, dtau: float, K: int) -> float:
    C = dtau * model.C_tilde
    det_C = np.linalg.det(C)
    lam_dt = model.lam * dtau

    bound = (np.exp(-(model.r + model.lam) * dtau)
             / (2.0 * np.pi * np.sqrt(det_C))
             * (np.e * lam_dt)**(K + 1) / (K + 1)**(K + 1))
    return bound
