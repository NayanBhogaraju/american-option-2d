from dataclasses import dataclass
import numpy as np


@dataclass
class Grid2D:
    x_in: np.ndarray
    y_in: np.ndarray
    x_dag: np.ndarray
    y_dag: np.ndarray
    x_ddag: np.ndarray
    y_ddag: np.ndarray
    dx: float
    dy: float
    dtau: float
    M: int
    N: int
    J: int

    @property
    def N_set(self):
        return np.arange(-self.N // 2 + 1, self.N // 2)

    @property
    def J_set(self):
        return np.arange(-self.J // 2 + 1, self.J // 2)


def build_grid(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    N: int, J: int, M: int, T: float,
) -> Grid2D:
    Px = x_max - x_min
    Py = y_max - y_min
    dx = Px / N
    dy = Py / J
    dtau = T / M

    x_hat0 = 0.5 * (x_min + x_max)
    y_hat0 = 0.5 * (y_min + y_max)

    x_dag_min = x_min - Px / 2.0
    x_dag_max = x_max + Px / 2.0
    y_dag_min = y_min - Py / 2.0
    y_dag_max = y_max + Py / 2.0

    x_ddag_min = -1.5 * Px
    x_ddag_max = 1.5 * Px
    y_ddag_min = -1.5 * Py
    y_ddag_max = 1.5 * Py

    N_dag = 2 * N
    J_dag = 2 * J
    N_ddag = 3 * N
    J_ddag = 3 * J

    x_in = x_hat0 + np.arange(-N // 2 + 1, N // 2) * dx
    y_in = y_hat0 + np.arange(-J // 2 + 1, J // 2) * dy

    x_dag = x_hat0 + np.arange(-N, N + 1) * dx
    y_dag = y_hat0 + np.arange(-J, J + 1) * dy

    x_ddag = np.arange(-(3 * N // 2 - 1), 3 * N // 2) * dx
    y_ddag = np.arange(-(3 * J // 2 - 1), 3 * J // 2) * dy

    return Grid2D(
        x_in=x_in, y_in=y_in,
        x_dag=x_dag, y_dag=y_dag,
        x_ddag=x_ddag, y_ddag=y_ddag,
        dx=dx, dy=dy, dtau=dtau,
        M=M, N=N, J=J,
    )
