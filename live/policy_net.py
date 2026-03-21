from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def _best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PolicyNet(nn.Module):

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, log_x: float, log_y: float) -> tuple[float, float]:
        device = next(self.parameters()).device
        inp = torch.tensor([[log_x, log_y]], dtype=torch.float32, device=device)
        out = self.forward(inp)
        pi_x = float(out[0, 0].cpu())
        pi_y = float(out[0, 1].cpu())
        return pi_x, pi_y

    @torch.no_grad()
    def predict_grid(
        self, log_x_arr: np.ndarray, log_y_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        Xm, Ym = np.meshgrid(log_x_arr, log_y_arr, indexing="ij")
        pts = np.stack([Xm.ravel(), Ym.ravel()], axis=1).astype(np.float32)
        inp = torch.from_numpy(pts).to(device)

        batch_size = 4096
        outputs = []
        for i in range(0, len(pts), batch_size):
            outputs.append(self.forward(inp[i : i + batch_size]).cpu().numpy())

        out = np.concatenate(outputs, axis=0)
        pi_x_grid = out[:, 0].reshape(Xm.shape)
        pi_y_grid = out[:, 1].reshape(Xm.shape)
        return pi_x_grid, pi_y_grid


def train_policy_net(
    net: PolicyNet,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    pi_x_grid: np.ndarray,
    pi_y_grid: np.ndarray,
    epochs: int = 400,
    lr: float = 3e-3,
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

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    net.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, tb in loader:
            optimizer.zero_grad()
            pred = net(xb)
            loss = -(tb * torch.log(pred + 1e-8)).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        scheduler.step()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            avg_loss = total_loss / len(dataset)
            print(f"  [PolicyNet] epoch {epoch:>4}/{epochs}  loss={avg_loss:.6f}")

    net.eval()
    return net


def save_policy_net(net: PolicyNet, path: str) -> None:
    torch.save(net.state_dict(), path)


def load_policy_net(path: str, hidden: int = 64) -> PolicyNet:
    net = PolicyNet(hidden=hidden)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net
