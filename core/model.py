from dataclasses import dataclass
import numpy as np


@dataclass
class MertonJumpDiffusion2D:
    sigma_x: float
    sigma_y: float
    rho: float
    lam: float
    mu_tilde_x: float
    mu_tilde_y: float
    sigma_tilde_x: float
    sigma_tilde_y: float
    rho_hat: float
    r: float
    T: float
    K: float

    def __post_init__(self):
        assert -1 < self.rho < 1, "Brownian correlation must be in (-1, 1)"
        assert -1 < self.rho_hat < 1, "Jump correlation must be in (-1, 1)"
        assert self.sigma_x > 0 and self.sigma_y > 0
        assert self.lam >= 0
        assert self.r > 0
        assert self.T > 0
        assert self.K > 0

    @property
    def kappa_x(self) -> float:
        return np.exp(self.mu_tilde_x + 0.5 * self.sigma_tilde_x**2) - 1.0

    @property
    def kappa_y(self) -> float:
        return np.exp(self.mu_tilde_y + 0.5 * self.sigma_tilde_y**2) - 1.0

    @property
    def C_tilde(self) -> np.ndarray:
        return np.array([
            [self.sigma_x**2, self.rho * self.sigma_x * self.sigma_y],
            [self.rho * self.sigma_x * self.sigma_y, self.sigma_y**2],
        ])

    @property
    def beta_tilde(self) -> np.ndarray:
        return np.array([
            self.r - self.lam * self.kappa_x - 0.5 * self.sigma_x**2,
            self.r - self.lam * self.kappa_y - 0.5 * self.sigma_y**2,
        ])

    @property
    def C_M(self) -> np.ndarray:
        return np.array([
            [self.sigma_tilde_x**2,
             self.rho_hat * self.sigma_tilde_x * self.sigma_tilde_y],
            [self.rho_hat * self.sigma_tilde_x * self.sigma_tilde_y,
             self.sigma_tilde_y**2],
        ])

    @property
    def mu_tilde_vec(self) -> np.ndarray:
        return np.array([self.mu_tilde_x, self.mu_tilde_y])


def case_I(X0: float = 90.0, Y0: float = 90.0) -> tuple:
    return MertonJumpDiffusion2D(
        sigma_x=0.12, sigma_y=0.15, rho=0.30,
        lam=0.60,
        mu_tilde_x=-0.10, mu_tilde_y=0.10,
        sigma_tilde_x=0.17, sigma_tilde_y=0.13,
        rho_hat=-0.20,
        r=0.05, T=1.0, K=100.0,
    ), X0, Y0


def case_II(X0: float = 40.0, Y0: float = 40.0) -> tuple:
    return MertonJumpDiffusion2D(
        sigma_x=0.30, sigma_y=0.30, rho=0.50,
        lam=2.0,
        mu_tilde_x=-0.50, mu_tilde_y=0.30,
        sigma_tilde_x=0.40, sigma_tilde_y=0.10,
        rho_hat=-0.60,
        r=0.05, T=0.5, K=40.0,
    ), X0, Y0


def case_III(X0: float = 40.0, Y0: float = 40.0) -> tuple:
    return MertonJumpDiffusion2D(
        sigma_x=0.20, sigma_y=0.30, rho=0.70,
        lam=8.0,
        mu_tilde_x=-0.05, mu_tilde_y=-0.20,
        sigma_tilde_x=0.45, sigma_tilde_y=0.06,
        rho_hat=0.50,
        r=0.05, T=1.0, K=40.0,
    ), X0, Y0
