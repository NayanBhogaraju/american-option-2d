"""
Live real-time two-asset allocation system.

Components
----------
data_feed   : yfinance historical + live price fetching
calibrator  : Merton jump-diffusion parameter estimation from returns
policy_net  : Tiny PyTorch MLP for real-time policy interpolation
system      : AllocationSystem orchestration (calibrate → solve → train → infer)
dashboard   : Streamlit real-time dashboard
"""

from .data_feed import DataFeed
from .calibrator import MertonCalibrator
from .policy_net import PolicyNet, train_policy_net
from .system import AllocationSystem

__all__ = [
    "DataFeed",
    "MertonCalibrator",
    "PolicyNet",
    "train_policy_net",
    "AllocationSystem",
]
