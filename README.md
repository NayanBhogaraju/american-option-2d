# Two-Asset Optimal Allocation System

A real-time portfolio allocation engine built on Merton jump-diffusion dynamics, solved via a discrete-time Bellman equation using 2D FFT convolution. Includes a live Streamlit dashboard, historical backtester with statistical significance testing, utility-indifference options pricing, and a residual RBF-KAN policy network.

---

## What it does

Given two assets (any Yahoo Finance tickers), the system:

1. **Calibrates** a bivariate Merton jump-diffusion model to 2 years of daily returns — estimating diffusion volatilities, Brownian correlation, jump intensity, jump size distributions, and real-world drifts with automatic shrinkage toward a CAPM prior
2. **Solves** the Bellman equation on a 128×128 log-price grid via FFT-based Green's function convolution, producing the theoretically optimal CRRA-utility-maximising allocation at every point in price space
3. **Trains** a Residual RBF-KAN policy network on the Bellman solution for microsecond inference at live prices
4. **Streams** live price updates (15-min delayed, yfinance free tier) and refreshes allocations every 15 minutes without re-running the full solve
5. **Backtests** the strategy over 5 years of history with rolling calibration, stationary block bootstrap confidence intervals, and a Sharpe ratio significance test
6. **Prices** protective puts using both Black-Scholes and a utility-indifference price derived from a second Bellman solve with put-modified terminal utility

---

## The Math

### Asset dynamics
Each asset follows a Merton jump-diffusion in log-price space:

$$\frac{dX}{X} = \mu_x \, dt + \sigma_x \, dW_x + (e^{J_x}-1)\,dN$$

where $N$ is a Poisson process with intensity $\lambda$, $(J_x, J_y)$ are correlated log-normal jumps, and $(W_x, W_y)$ are Brownian motions with correlation $\rho$.

### Bellman equation
The investor rebalances at $M$ equally-spaced dates. The optimal value function satisfies:

$$V(x, y, t_m) = \max_{\pi \in \Pi} \sum_{l,d} g_{\pi}(x - x_l,\, y - y_d) \cdot V(x_l, y_d, t_{m+1})$$

where the modified Green's kernel weights future states by both transition probability and portfolio return:

$$g_{\pi}(\Delta x, \Delta y) = g(\Delta x, \Delta y) \cdot \bigl[\pi_x e^{\Delta x} + \pi_y e^{\Delta y} + \pi_c e^{r\Delta\tau}\bigr]^{\gamma}$$

### FFT convolution
Since $g_{\pi}$ depends only on displacement, the Bellman sum is a 2D convolution — computed via FFT in $O(NJ \log NJ)$ per time step. Green's function FFTs are precomputed once per policy at startup.

### CRRA utility
Terminal utility is $U(W) = W^{\gamma} / \gamma$ evaluated on a basket $W_T = \alpha e^{x_T} + (1-\alpha)e^{y_T}$. Risk aversion $\gamma < 0$; more negative = more conservative.

### Drift shrinkage (anti-lookback-bias)
The sample mean drift is blended toward a CAPM prior via time-series cross-validation. The optimal shrinkage weight $\lambda^{\ast}$ is selected automatically by maximising the predictive log-likelihood of held-out returns — no user input required.

$$\hat{\mu}_x = (1-\lambda^{\ast})\,\bar{\mu}_x^{\text{sample}} + \lambda^{\ast}\,(r + \beta_x \cdot \text{ERP})$$

$$\lambda^{\ast} = \underset{\lambda \in [0,1]}{\arg\max} \sum_{t \in \text{val}} \log p\!\left(r_t \mid \hat{\mu}(\lambda),\, \sigma\right)$$

### Residual RBF-KAN policy network
The Bellman solution is distilled into a 2→4→3 Kolmogorov–Arnold Network with Gaussian RBF basis functions. Each edge learns a non-linear univariate function; a learnable residual baseline $b \in \mathbb{R}^3$ captures the near-flat Merton structure:

$$\text{KAN}_{\text{out}}(x,y) = \text{softmax}\!\left(b + \text{KAN}_2\!\left(\text{SiLU}\!\left(\text{KAN}_1(x,y)\right)\right)\right)$$

~300 parameters vs ~4,500 for a comparable MLP. Trained with MSE loss and AdamW + cosine annealing.

### Utility-indifference options pricing
A protective put on the X asset is priced by finding $p^{\ast}$ such that the investor is indifferent between paying $p^{\ast}$ for the put and going unhedged. The put floors the X component of the basket at the normalised strike $k = K/S_{x,0}$, giving a modified terminal utility:

$$U_{\text{put}}(x, y) = \!\left(\alpha \cdot \max(e^x, k) + (1-\alpha)\cdot e^y\right)^{\!\gamma} \!\big/ \gamma$$

By CRRA scale invariance the indifference price is:

$$p^{\ast} = W_0 \cdot \left(1 - \left(\frac{\bar{V}_{\text{no put}}}{\bar{V}_{\text{with put}}}\right)^{\!1/\gamma}\right)$$

where $\bar{V}$ is the normalised value at the initial state from each Bellman solve. The Black-Scholes price is shown alongside as a benchmark.

### Bootstrap confidence intervals
Backtest Sharpe ratios and returns are reported with 95% confidence intervals using a stationary block bootstrap (block length = 21 trading days, 500 iterations). The Sharpe significance test reports the bootstrapped $p$-value for $H_0$: Sharpe(optimal) $\leq$ Sharpe(50/50).

---

## Project Structure

```
american_option_2d/
├── core/
│   ├── model.py              # MertonJumpDiffusion2D dataclass
│   ├── greens_function.py    # Poisson-series Green's function
│   ├── grid.py               # Spatial/temporal grid
│   ├── pricer.py             # AmericanOptionPricer2D (FFT convolution)
│   ├── allocator.py          # TwoAssetAllocator (Bellman solver)
│   └── payoffs.py            # Payoff functions
├── live/
│   ├── dashboard.py          # Landing page (Streamlit entry point)
│   ├── pages/
│   │   ├── 1_Model.py        # Live allocation dashboard + options overlay
│   │   └── 2_Backtest.py     # Historical backtest + bootstrap CI + significance tests
│   ├── system.py             # AllocationSystem orchestrator
│   ├── calibrator.py         # MertonCalibrator with auto drift shrinkage
│   ├── data_feed.py          # yfinance data feed
│   ├── policy_net.py         # Residual RBF-KAN (PyTorch, MPS-accelerated)
│   ├── backtest.py           # Rolling-window backtester + bootstrap_ci()
│   └── options_overlay.py    # BS + utility-indifference put pricing
├── examples/
│   ├── run_convergence_study.py
│   ├── run_comprehensive_test.py
│   ├── plot_early_exercise.py
│   └── run_domain_sensitivity.py
├── tests/
│   └── test_greens.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option pricing only (core library)
```bash
pip install -r requirements.txt
python examples/run_convergence_study.py
```

### Live allocation dashboard
```bash
pip install -r requirements.txt
streamlit run live/dashboard.py
```

The dashboard opens at `http://localhost:8501`. The landing page has a full user guide. Use the sidebar to navigate:
- **Model** — run calibration, view live allocation, policy surfaces, and options overlay
- **Backtest** — rolling historical backtest with bootstrap confidence intervals and significance testing

---

## Hardware

Designed for **Apple M4 Air (16 GB unified memory)**. PyTorch automatically uses the MPS backend when available.

| Grid | Memory | Time (M4 Air) |
|------|--------|---------------|
| N=J=64, M=20, n_pi=11 | ~130 MB | ~30 s |
| N=J=128, M=20, n_pi=11 | ~530 MB | ~3 min |
| N=J=192, M=20, n_pi=11 | ~1.2 GB | ~10 min |

Second Bellman solve for indifference pricing uses N=J=64, M=5 (~5–10 s).

---

## Dashboard Features

| Feature | Description |
|---|---|
| Live prices | 15-min delayed yfinance feed, auto-refreshes every 15 min |
| Allocation gauges | Optimal π_x, π_y, cash as live speedometers |
| Policy surface | 3 heatmaps showing optimal allocation across all price scenarios |
| P&L tracker | Dollar P&L since last calibration vs 50/50 benchmark |
| Expected returns | Annualised return comparison: optimal vs 50/50 vs all-X vs cash |
| Calibration panel | σ, ρ, λ, blended μ, auto shrinkage λ*, CAPM betas |
| Equity premium | Why each asset is / isn't allocated to |
| Options overlay | BS price + utility-indifference price side-by-side |
| Backtest page | 5-year rolling calibration, wealth chart, performance table |
| Bootstrap CI | 95% confidence intervals on Sharpe, ann. return, max drawdown |
| Sharpe significance | Bootstrapped p-value test: optimal vs 50/50 |

---

## Suggested Asset Pairs

| Pair | Correlation | What it tests |
|---|---|---|
| SPY / QQQ | ~0.92 | Baseline — nearly 1D allocation |
| SPY / GLD | ~0.05 | Low correlation — genuine 2D spatial structure |
| QQQ / TLT | ~−0.30 | Negative correlation — natural hedge pair |
| BTC-USD / ETH-USD | ~0.85 | Crypto correlation |

---

## References

- Merton, R.C. (1969). Lifetime portfolio selection under uncertainty: the continuous-time case. *Review of Economics and Statistics*, 51(3), 247–257.
- Zhou, H. & Dang, D.-M. (2025). Numerical analysis of American option pricing in a two-asset jump-diffusion model. *Applied Numerical Mathematics*, 216, 98–126.
- Liu, Z., Wang, Y., Vaidya, S., et al. (2024). KAN: Kolmogorov–Arnold Networks. *arXiv:2404.19756*.
- James, W. & Stein, C. (1961). Estimation with quadratic loss. *Proceedings of the Fourth Berkeley Symposium*, 1, 361–379.
- Hodges, S. & Neuberger, A. (1989). Optimal replication of contingent claims under transaction costs. *Review of Futures Markets*, 8(2), 222–239.

---

## License

Educational and research use. See referenced papers for full attribution.
