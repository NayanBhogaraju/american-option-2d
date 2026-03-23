from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SURROGATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate_kan.pt")
_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "surrogate_kan_meta.json")

# Normalization ranges [lo, hi] for each of the 9 inputs:
# sigma_x, sigma_y, rho, lam, mu_x, mu_y, r, dx, dy
PARAM_RANGES = np.array([
    [0.05, 0.80],   # sigma_x
    [0.05, 0.80],   # sigma_y
    [-0.90, 0.90],  # rho
    [0.0,  25.0],   # lam (jump intensity / yr)
    [-0.35, 0.50],  # mu_x (real-world drift)
    [-0.35, 0.50],  # mu_y
    [0.005, 0.10],  # r (risk-free rate)
    [-0.80, 0.80],  # dx (log-price displacement X from calibration center)
    [-0.80, 0.80],  # dy
], dtype=np.float64)

_MID  = (PARAM_RANGES[:, 0] + PARAM_RANGES[:, 1]) / 2.0
_HALF = (PARAM_RANGES[:, 1] - PARAM_RANGES[:, 0]) / 2.0


def normalize_input(
    sigma_x: float, sigma_y: float, rho: float, lam: float,
    mu_x: float, mu_y: float, r: float,
    dx: float = 0.0, dy: float = 0.0,
) -> np.ndarray:
    v = np.array([sigma_x, sigma_y, rho, lam, mu_x, mu_y, r, dx, dy], dtype=np.float64)
    return ((v - _MID) / _HALF).astype(np.float32)


def _normalize_batch(arr: np.ndarray) -> np.ndarray:
    return ((arr - _MID) / _HALF).astype(np.float32)


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _RBFKANLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 12,
                 grid_range: tuple[float, float] = (-3.0, 3.0)):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(in_dim, out_dim, grid_size))
        self.base  = nn.Linear(in_dim, out_dim, bias=True)
        centers = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("centers", centers)
        h = (grid_range[1] - grid_range[0]) / max(grid_size - 1, 1)
        self.register_buffer("bandwidth", torch.tensor(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff  = x.unsqueeze(-1) - self.centers
        basis = torch.exp(-0.5 * (diff / self.bandwidth) ** 2)
        spline = torch.einsum("big,iog->bo", basis, self.coeff)
        return spline + self.base(x)


class SurrogateKAN(nn.Module):
    """
    Parameter-conditioned RBF-KAN.
    Input:  9D — (sigma_x, sigma_y, rho, lam, mu_x, mu_y, r, dx, dy) normalized to [-1,1]
    Output: (pi_x, pi_y, pi_cash) via softmax

    Generalises across all calibration parameter regimes so the Bellman equation
    only needs to be solved offline during training, not at every backtest step.
    """

    def __init__(self, hidden1: int = 24, hidden2: int = 12, grid_size: int = 12):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(3))
        self.kan1 = _RBFKANLayer(9,       hidden1, grid_size)
        self.kan2 = _RBFKANLayer(hidden1, hidden2, grid_size)
        self.kan3 = _RBFKANLayer(hidden2, 3,       grid_size)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1   = self.act(self.kan1(x))
        h2   = self.act(self.kan2(h1))
        corr = self.kan3(h2)
        return torch.softmax(self.base + corr, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        sigma_x: float, sigma_y: float, rho: float, lam: float,
        mu_x: float, mu_y: float, r: float,
        dx: float = 0.0, dy: float = 0.0,
    ) -> tuple[float, float]:
        device = next(self.parameters()).device
        inp = normalize_input(sigma_x, sigma_y, rho, lam, mu_x, mu_y, r, dx, dy)
        t   = torch.from_numpy(inp).unsqueeze(0).to(device)
        out = self.forward(t)
        pi_x   = float(out[0, 0].cpu())
        pi_y   = float(out[0, 1].cpu())
        total  = pi_x + pi_y
        if total > 1.0:
            pi_x /= total
            pi_y /= total
        return pi_x, pi_y


def generate_training_data(
    n_solves: int,
    gamma: float,
    alpha: float,
    horizon_years: float,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute n_solves Bellman solutions on a fast grid (N=32, M=5) spanning
    the realistic parameter space.  Returns (inputs [N,9], targets [N,3]).

    px = py = 1.0 for all solves so the grid is centered at log-price (0,0),
    and the x/y grid values ARE the displacement (dx, dy) directly.
    """
    from core.model import MertonJumpDiffusion2D
    from core.allocator import TwoAssetAllocator, crra_basket_utility

    rng      = np.random.default_rng(42)
    terminal = crra_basket_utility(gamma, alpha=alpha)

    params = rng.uniform(
        PARAM_RANGES[:7, 0], PARAM_RANGES[:7, 1], size=(n_solves, 7)
    )
    sigma_tilde_arr = rng.uniform(0.02, 0.12, size=(n_solves, 2))

    all_inputs:  list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for i in range(n_solves):
        sigma_x, sigma_y, rho, lam, mu_x, mu_y, r = params[i]
        sigma_tx, sigma_ty = sigma_tilde_arr[i]

        try:
            model = MertonJumpDiffusion2D(
                sigma_x=float(sigma_x), sigma_y=float(sigma_y),
                rho=float(rho), lam=float(lam),
                mu_tilde_x=-0.02, mu_tilde_y=-0.02,
                sigma_tilde_x=float(sigma_tx), sigma_tilde_y=float(sigma_ty),
                rho_hat=0.0, r=float(r),
                T=float(horizon_years), K=1.0,
            )
            allocator = TwoAssetAllocator(
                model, 1.0, 1.0,
                gamma=gamma,
                mu_x_real=float(mu_x),
                mu_y_real=float(mu_y),
                allow_short=False,
                domain_half_width_x=0.8,
                domain_half_width_y=0.8,
                N=32, J=32, M=5, n_pi=11,
            )
            res = allocator.solve(terminal, return_grid_values=True, return_policy=True)
        except Exception:
            if progress_cb:
                progress_cb(i + 1, n_solves)
            continue

        x_grid    = res["x"]
        y_grid    = res["y"]
        pi_x_grid = res["pi_x"]
        pi_y_grid = res["pi_y"]

        Xm, Ym   = np.meshgrid(x_grid, y_grid, indexing="ij")
        dx_flat  = Xm.ravel()
        dy_flat  = Ym.ravel()
        pix_flat = pi_x_grid.ravel()
        piy_flat = pi_y_grid.ravel()
        pic_flat = np.clip(1.0 - pix_flat - piy_flat, 0.0, 1.0)

        n_pts       = len(dx_flat)
        param_block = np.tile([sigma_x, sigma_y, rho, lam, mu_x, mu_y, r], (n_pts, 1))
        inp_block   = np.concatenate(
            [param_block, dx_flat[:, None], dy_flat[:, None]], axis=1
        ).astype(np.float32)

        tgt_raw  = np.stack([pix_flat, piy_flat, pic_flat], axis=1)
        row_sums = tgt_raw.sum(axis=1, keepdims=True)
        tgt_norm = (tgt_raw / np.maximum(row_sums, 1e-8)).astype(np.float32)

        all_inputs.append(inp_block)
        all_targets.append(tgt_norm)

        if progress_cb:
            progress_cb(i + 1, n_solves)

    if not all_inputs:
        raise RuntimeError("No successful Bellman solves — check model parameters.")

    return np.concatenate(all_inputs, axis=0), np.concatenate(all_targets, axis=0)


def train_surrogate(
    net: SurrogateKAN,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int = 300,
    lr: float = 3e-3,
    batch_size: int = 4096,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[SurrogateKAN, float, float]:
    if device is None:
        device = _best_device()

    inp_t = torch.from_numpy(_normalize_batch(inputs)).to(device)
    tgt_t = torch.from_numpy(targets).to(device)

    dataset = TensorDataset(inp_t, tgt_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    net.train()
    first_mse = final_mse = 0.0
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, tb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(net(xb), tb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        scheduler.step()
        avg = total / len(dataset)
        if epoch == 1:
            first_mse = avg
        final_mse = avg
        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  [SurrogateKAN] epoch {epoch:>4}/{epochs}  mse={avg:.6f}")

    net.eval()
    return net, first_mse, final_mse


def save_surrogate(
    net: SurrogateKAN,
    gamma: float, alpha: float, horizon_years: float,
    n_solves: int, train_mse: float,
    path: str = _SURROGATE_PATH,
    meta_path: str = _META_PATH,
) -> None:
    torch.save(net.state_dict(), path)
    with open(meta_path, "w") as f:
        json.dump({
            "gamma": gamma, "alpha": alpha, "horizon_years": horizon_years,
            "n_solves": n_solves, "train_mse": train_mse,
            "hidden1": 24, "hidden2": 12,
        }, f, indent=2)


def load_surrogate(
    path: str = _SURROGATE_PATH,
    meta_path: str = _META_PATH,
) -> tuple[SurrogateKAN, dict]:
    with open(meta_path) as f:
        meta = json.load(f)
    net = SurrogateKAN(hidden1=meta["hidden1"], hidden2=meta["hidden2"])
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net, meta


def surrogate_available(
    path: str = _SURROGATE_PATH,
    meta_path: str = _META_PATH,
) -> bool:
    return os.path.exists(path) and os.path.exists(meta_path)


def validate_surrogate(
    net: SurrogateKAN,
    gamma: float,
    alpha: float,
    horizon_years: float,
    tol: float = 0.05,
) -> tuple[bool, float]:
    """
    Run one exact Bellman solve at a representative parameter point and compare
    allocations to the surrogate.  Returns (passed, max_abs_error).

    Raises RuntimeError if max_abs_error > tol (default 5pp).
    """
    from core.model import MertonJumpDiffusion2D
    from core.allocator import TwoAssetAllocator, crra_basket_utility

    terminal = crra_basket_utility(gamma, alpha=alpha)
    model = MertonJumpDiffusion2D(
        sigma_x=0.20, sigma_y=0.18, rho=0.40, lam=2.0,
        mu_tilde_x=-0.02, mu_tilde_y=-0.02,
        sigma_tilde_x=0.05, sigma_tilde_y=0.05,
        rho_hat=0.0, r=0.045,
        T=float(horizon_years), K=1.0,
    )
    allocator = TwoAssetAllocator(
        model, 1.0, 1.0,
        gamma=gamma,
        mu_x_real=0.08,
        mu_y_real=0.06,
        allow_short=False,
        domain_half_width_x=0.8,
        domain_half_width_y=0.8,
        N=32, J=32, M=5, n_pi=11,
    )
    res = allocator.solve(terminal, return_grid_values=True, return_policy=True)

    # Evaluate exact policy at centre point (dx=0, dy=0)
    from scipy.interpolate import RegularGridInterpolator
    kw = dict(method="linear", bounds_error=False, fill_value=None)
    ix = RegularGridInterpolator((res["x"], res["y"]), res["pi_x"], **kw)
    iy = RegularGridInterpolator((res["x"], res["y"]), res["pi_y"], **kw)
    exact_pix = float(np.clip(ix([[0.0, 0.0]]).item(), 0, 1))
    exact_piy = float(np.clip(iy([[0.0, 0.0]]).item(), 0, 1))
    tot = exact_pix + exact_piy
    if tot > 1.0:
        exact_pix /= tot
        exact_piy /= tot

    surr_pix, surr_piy = net.predict(
        sigma_x=0.20, sigma_y=0.18, rho=0.40, lam=2.0,
        mu_x=0.08, mu_y=0.06, r=0.045,
        dx=0.0, dy=0.0,
    )

    err = max(abs(surr_pix - exact_pix), abs(surr_piy - exact_piy))
    passed = err <= tol
    if not passed:
        raise RuntimeError(
            f"Surrogate validation failed: max allocation error {err*100:.1f}pp > {tol*100:.0f}pp tolerance. "
            f"Exact π_x={exact_pix:.3f} π_y={exact_piy:.3f} vs "
            f"Surrogate π_x={surr_pix:.3f} π_y={surr_piy:.3f}. "
            "Rebuild the surrogate with more training solves."
        )
    return passed, err
