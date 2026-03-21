import numpy as np


def put_on_min_payoff(x: np.ndarray, y: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - np.minimum(np.exp(x), np.exp(y)), 0.0)


def put_on_average_payoff(x: np.ndarray, y: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - 0.5 * (np.exp(x) + np.exp(y)), 0.0)


def put_on_max_payoff(x: np.ndarray, y: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - np.maximum(np.exp(x), np.exp(y)), 0.0)


def spread_option_payoff(x: np.ndarray, y: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - (np.exp(x) - np.exp(y)), 0.0)
