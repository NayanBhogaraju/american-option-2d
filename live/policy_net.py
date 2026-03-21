from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _RBFKANLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int = 12,
        grid_range: tuple[float, float] = (-4.0, 4.0),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        self.coeff = nn.Parameter(
            torch.zeros(in_dim, out_dim, grid_size)
        )
        self.base = nn.Linear(in_dim, out_dim, bias=True)

        centers = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("centers", centers)
        h = (grid_range[1] - grid_range[0]) / max(grid_size - 1, 1)
        self.register_buffer("bandwidth", torch.tensor(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-1) - self.centers
        basis = torch.exp(-0.5 * (diff / self.bandwidth) ** 2)
        spline = torch.einsum("big,iog->bo", basis, self.coeff)
        return spline + self.base(x)


class PolicyNet(nn.Module):

    def __init__(
        self,
        hidden: int = 4,
        grid_size: int = 12,
        grid_range: tuple[float, float] = (-4.0, 4.0),
    ):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(3))
        self.kan1 = _RBFKANLayer(2, hidden, grid_size, grid_range)
        self.kan2 = _RBFKANLayer(hidden, 3, grid_size, grid_range)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.kan1(x))
        correction = self.kan2(h)
        return torch.softmax(self.base + correction, dim=-1)

    @torch.no_grad()
    def predict(self, log_x: float, log_y: float) -> tuple[float, float]:
        device = next(self.parameters()).device
        inp = torch.tensor([[log_x, log_y]], dtype=torch.float32, device=device)
        out = self.forward(inp)
        return float(out[0, 0].cpu()), float(out[0, 1].cpu())

    @torch.no_grad()
    def predict_grid(
        self, log_x_arr: np.ndarray, log_y_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        Xm, Ym = np.meshgrid(log_x_arr, log_y_arr, indexing="ij")
        pts = np.stack([Xm.ravel(), Ym.ravel()], axis=1).astype(np.float32)
        inp = torch.from_numpy(pts).to(device)

        outputs = []
        for i in range(0, len(pts), 4096):
            outputs.append(self.forward(inp[i: i + 4096]).cpu().numpy())

        out = np.concatenate(outputs, axis=0)
        return out[:, 0].reshape(Xm.shape), out[:, 1].reshape(Xm.shape)


def train_policy_net(
    net: PolicyNet,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pi_x_grid: np.ndarray,
    pi_y_grid: np.ndarray,
    epochs: int = 200,
    lr: float = 5e-3,
    batch_size: int = 1024,
    device: torch.device | None = None,
    verbose: bool = True,
) -> PolicyNet:
    if device is None:
        device = _best_device()

    net = net.to(device)

    Xm, Ym = np.meshgrid(x_grid, y_grid, indexing="ij")
    inputs = np.stack([Xm.ravel(), Ym.ravel()], axis=1).astype(np.float32)

    pi_x_flat = pi_x_grid.ravel().astype(np.float32)
    pi_y_flat = pi_y_grid.ravel().astype(np.float32)
    pi_cash_flat = np.clip(1.0 - pi_x_flat - pi_y_flat, 0.0, 1.0)

    targets_raw = np.stack([pi_x_flat, pi_y_flat, pi_cash_flat], axis=1)
    row_sums = targets_raw.sum(axis=1, keepdims=True)
    targets = (targets_raw / np.maximum(row_sums, 1e-8)).astype(np.float32)

    inp_t = torch.from_numpy(inputs).to(device)
    tgt_t = torch.from_numpy(targets).to(device)

    dataset = TensorDataset(inp_t, tgt_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    net.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, tb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = F.mse_loss(pred, tb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        scheduler.step()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            avg_loss = total_loss / len(dataset)
            print(f"  [PolicyNet/KAN] epoch {epoch:>4}/{epochs}  mse={avg_loss:.6f}")

    net.eval()
    return net


def save_policy_net(net: PolicyNet, path: str) -> None:
    torch.save(net.state_dict(), path)


def load_policy_net(path: str, hidden: int = 4, grid_size: int = 12) -> PolicyNet:
    net = PolicyNet(hidden=hidden, grid_size=grid_size)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net
