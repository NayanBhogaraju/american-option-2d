import streamlit as st

st.set_page_config(
    page_title="Two-Asset Allocation System",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Two-Asset Optimal Allocation System")
st.caption("Merton Jump-Diffusion · Bellman Equation · CRRA Utility · Live Market Data")

st.markdown("---")


# ── What it is ────────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### What it is")
    st.markdown(
        "A mathematically rigorous portfolio optimizer that finds the "
        "**theoretically optimal split** between two assets and cash "
        "at every point in price space — not just at today's prices, "
        "but across all possible future price scenarios simultaneously."
    )
with c2:
    st.markdown("#### What it does")
    st.markdown(
        "Each time you run it: fetches 2 years of market data, fits a "
        "Merton jump-diffusion model to both assets, solves a "
        "Bellman equation on a 128×128 price grid via FFT convolution, "
        "then trains a **Residual KAN** (Kolmogorov-Arnold Network) on the "
        "solution for microsecond inference — giving you live allocation "
        "recommendations updated every 15 minutes."
    )
with c3:
    st.markdown("#### What makes it different")
    st.markdown(
        "Most allocators use Markowitz (static, one-shot). This one "
        "is **dynamic** — it accounts for how assets jump, how they "
        "co-move, and how the optimal allocation changes as prices "
        "evolve. It also automatically corrects for look-back bias "
        "in drift estimates using time-series cross-validation."
    )

st.markdown("---")


# ── Math ──────────────────────────────────────────────────────────────────────

st.markdown("## How It Works — The Math")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Asset Model", "Bellman Equation", "Green's Function", "CRRA Utility", "Drift Shrinkage", "KAN Policy Net"
])

with tab1:
    st.markdown("### Merton Jump-Diffusion (2D)")
    st.markdown(
        "Each asset follows a geometric Brownian motion with superimposed Poisson jumps. "
        "For two correlated assets X and Y:"
    )
    st.latex(r"""
        \frac{dX}{X} = \mu_x \, dt + \sigma_x \, dW_x + (e^{J_x} - 1) \, dN
    """)
    st.latex(r"""
        \frac{dY}{Y} = \mu_y \, dt + \sigma_y \, dW_y + (e^{J_y} - 1) \, dN
    """)
    st.markdown(
        "where $dW_x, dW_y$ are correlated Brownian motions with $\\rho = \\text{Corr}(dW_x, dW_y)$, "
        "$N$ is a Poisson process with intensity $\\lambda$ (jumps per year), "
        "and $(J_x, J_y)$ are correlated log-normal jump sizes. "
        "Working in log-price space $x = \\log X$, $y = \\log Y$, the increments are Gaussian between jumps."
    )
    m1, m2 = st.columns(2)
    m1.markdown(
        "**Calibrated parameters**\n"
        "- $\\sigma_x, \\sigma_y$ — diffusion volatilities\n"
        "- $\\rho$ — Brownian correlation\n"
        "- $\\lambda$ — jump intensity (events/yr)\n"
    )
    m2.markdown(
        "- $\\tilde{\\mu}_x, \\tilde{\\mu}_y$ — mean log-jump sizes\n"
        "- $\\tilde{\\sigma}_x, \\tilde{\\sigma}_y$ — jump volatilities\n"
        "- $\\hat{\\rho}$ — jump correlation\n"
        "- $r$ — risk-free rate (13-week T-bill)\n"
    )

with tab2:
    st.markdown("### Discrete-Time Bellman Equation")
    st.markdown(
        "The investor rebalances at $M$ equally-spaced dates over horizon $T$. "
        "At each date the investor picks allocations $(\\pi_x, \\pi_y, \\pi_c)$ "
        "— fractions in asset X, asset Y, and cash — subject to $\\pi_x + \\pi_y + \\pi_c = 1$, $\\pi \\geq 0$."
    )
    st.latex(r"""
        V(x, y, t_m) = \max_{\pi \in \Pi} \;
        \sum_{l,d} \; g_\pi(x - x_l,\; y - y_d) \cdot V(x_l, y_d, t_{m+1})
    """)
    st.markdown(
        "The modified Green's function kernel $g_\\pi$ weights each future state by "
        "both its probability (from the Merton transition density) and the portfolio "
        "return it generates:"
    )
    st.latex(r"""
        g_\pi(\Delta x, \Delta y) \;=\;
        g(\Delta x, \Delta y) \cdot
        \bigl[\pi_x e^{\Delta x} + \pi_y e^{\Delta y} + \pi_c e^{r\Delta\tau}\bigr]^\gamma
    """)
    st.markdown(
        "The $\\max$ over $\\Pi$ replaces the early-exercise $\\max$ of option pricing — "
        "same algorithm, different economic interpretation. "
        "Terminal condition: $V(x,y,T) = U(\\alpha e^x + (1-\\alpha)e^y)$."
    )

with tab3:
    st.markdown("### Green's Function via 2D FFT")
    st.markdown(
        "The key computational insight: $g_\\pi(\\Delta x, \\Delta y)$ depends only on the "
        "*displacement* $(\\Delta x, \\Delta y)$, not on the absolute price level. "
        "This means the Bellman sum is a **2D discrete convolution**, computable via FFT:"
    )
    st.latex(r"""
        (g_\pi \star V)(x, y) \;=\; \mathcal{F}^{-1}\!\bigl[\hat{g}_\pi \cdot \hat{V}\bigr](x, y)
    """)
    st.markdown(
        "For each candidate policy $\\pi$, $\\hat{g}_\\pi$ is precomputed **once** at the start. "
        "Then each Bellman step costs one FFT per policy per time step — $O(NJ \\log NJ)$ "
        "rather than $O(N^2 J^2)$ for naive quadrature. "
        "For $N = J = 128$ and 11 policies, this runs in ~3 minutes on an M4 Air."
    )
    st.markdown(
        "The Merton Green's function is computed via a truncated Poisson series — "
        "the $k$-jump term is a bivariate Gaussian, summed until the tail probability is below $10^{-12}$."
    )

with tab4:
    st.markdown("### CRRA Utility and Risk Aversion")
    st.markdown(
        "The investor maximises expected **Constant Relative Risk Aversion** utility "
        "of terminal wealth:"
    )
    st.latex(r"""
        U(W) = \frac{W^\gamma}{\gamma}, \qquad \gamma < 1,\; \gamma \neq 0
    """)
    st.markdown("Terminal wealth is measured as a basket of the two assets:")
    st.latex(r"""
        W_T = \alpha \, e^{x_T} + (1-\alpha) \, e^{y_T}
    """)
    mc1, mc2 = st.columns(2)
    mc1.markdown(
        "**Risk aversion γ**\n"
        "- $\\gamma = -0.1$ → near risk-neutral, aggressive\n"
        "- $\\gamma = -1$ → moderate (log-utility proxy)\n"
        "- $\\gamma = -5$ → very conservative\n"
        "\nMore negative → smaller equity allocations, larger cash buffer."
    )
    mc2.markdown(
        "**Risky vs cash condition**\n\n"
        "An asset beats cash when its equity premium clears the CRRA hurdle:\n"
    )
    mc2.latex(r"\mu_{\text{real}} - r \;>\; r + \tfrac{3}{2}\sigma^2")
    mc2.markdown("_(approximate, valid near γ = −1)_")

with tab5:
    st.markdown("### Automatic Drift Shrinkage (Anti-Lookback-Bias)")
    st.markdown(
        "The drift $\\mu_{\\text{real}}$ is the hardest parameter to estimate — "
        "the sample mean is swamped by noise and sensitive to recent regimes. "
        "A raw 5-year estimate of GLD drift can reach +32% due to the gold rally, "
        "which is not a reliable forecast."
    )
    st.markdown("The model automatically blends the sample mean toward a CAPM prior:")
    st.latex(r"""
        \hat{\mu}_x \;=\; (1-\lambda^*)\,\bar{\mu}_x^{\text{sample}} \;+\; \lambda^*\,\mu_x^{\text{CAPM}}
    """)
    st.latex(r"""
        \mu_x^{\text{CAPM}} \;=\; r + \beta_x \cdot \text{ERP}, \qquad
        \beta_x = \frac{\text{Cov}(r_x,\, r_m)}{\text{Var}(r_m)}
    """)
    st.markdown(
        "The optimal $\\lambda^*$ is chosen automatically via **time-series cross-validation**: "
        "the last 20% of the calibration window is held out, and $\\lambda^*$ is the value "
        "from a 41-point grid that maximises the predictive log-likelihood of the held-out returns. "
        "No user input required — the data decides."
    )
    st.latex(r"""
        \lambda^* = \underset{\lambda \in [0,1]}{\arg\max} \;
        \sum_{t \in \text{val}} \log p\!\left(r_t \;\middle|\; \hat{\mu}(\lambda),\, \sigma\right)
    """)

with tab6:
    st.markdown("### Residual KAN Policy Network")
    st.markdown(
        "The Bellman solver produces a discrete policy grid — optimal allocations at every "
        "$(x, y)$ grid point. A neural network is trained on this grid so that inference "
        "at live prices is instantaneous, without re-running the solver."
    )

    st.markdown("#### Why KAN instead of MLP?")
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(
            "**Old architecture (MLP)**\n"
            "- 2 → 64 → 64 → 3, SiLU activations\n"
            "- ~4,500 parameters\n"
            "- Cross-entropy loss (designed for classification)\n"
            "- Learns the full allocation from scratch\n"
            "- 400 epochs to converge\n"
            "- Softmax output (treats cash symmetrically with equities)\n"
        )
    with k2:
        st.markdown(
            "**New architecture (Residual KAN)**\n"
            "- 2 → 4 → 3 with RBF basis on each edge\n"
            "- ~300 parameters — 15× smaller\n"
            "- MSE loss (correct for continuous regression)\n"
            "- Residual baseline absorbs the dominant constant policy\n"
            "- 150 epochs to converge\n"
            "- Structured softmax output with learned constant offset\n"
        )

    st.markdown("#### Architecture")
    st.markdown(
        "The network has two components that add together before the final softmax:"
    )
    st.latex(r"""
        \pi(x, y) \;=\; \text{softmax}\!\Bigl(\underbrace{\mathbf{b}}_{\text{residual base}} \;+\; \underbrace{\text{KAN}(x,\, y)}_{\text{spatial correction}}\Bigr)
    """)
    st.markdown(
        "**Residual base** $\\mathbf{b} \\in \\mathbb{R}^3$ is a learnable constant vector. "
        "It quickly converges to the dominant allocation (e.g. [0.7, 0.0, 0.3]) in the first few epochs, "
        "so the KAN only needs to learn the small spatial deviation from that baseline."
    )

    st.markdown("#### KAN layers — RBF basis functions")
    st.markdown(
        "In a standard MLP, each node applies a fixed activation function (e.g. SiLU) to a weighted sum of inputs. "
        "In a KAN, each **edge** $(i \\to j)$ has its own learned univariate function $\\varphi_{ij}$:"
    )
    st.latex(r"""
        \text{KAN layer output}_j \;=\; \sum_{i} \varphi_{ij}(x_i)
        \;+\; \sum_{i} w_{ij}^{\text{base}} \, x_i
    """)
    st.markdown(
        "Each $\\varphi_{ij}$ is parameterised as a weighted sum of **Gaussian RBF basis functions** "
        "centred on a fixed grid of 12 points over $[-4, 4]$:"
    )
    st.latex(r"""
        \varphi_{ij}(x) \;=\; \sum_{k=1}^{12} c_{ijk} \exp\!\left(-\frac{(x - \mu_k)^2}{2h^2}\right)
    """)
    st.markdown(
        "where $\\mu_k$ are evenly spaced grid centres and $h$ is the inter-centre spacing. "
        "The coefficients $c_{ijk}$ are learned during training. "
        "This is mathematically equivalent to B-spline KANs (Liu et al., 2024) but implemented "
        "natively in PyTorch without the `pykan` dependency."
    )

    st.markdown("#### Why this matters for a near-flat policy surface")
    st.markdown(
        "From the heatmaps you can see that the CRRA optimal policy is nearly constant across prices "
        "(Merton's theorem: the optimal fraction is price-independent for CRRA + GBM). "
        "An MLP wastes most of its capacity learning to be constant. "
        "The residual base handles this exactly, and the KAN's per-edge learned functions "
        "capture the small boundary-driven deviations with far fewer parameters."
    )

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Parameters", "~300", delta="-4,247 vs MLP", delta_color="normal")
    kc2.metric("Training epochs", "150", delta="-250 vs MLP", delta_color="normal")
    kc3.metric("Loss function", "MSE", delta="vs cross-entropy", delta_color="off")

st.markdown("---")


# ── Usage guide ───────────────────────────────────────────────────────────────

st.markdown("## How to Use It")

st.markdown("### Step 1 — Go to the Model page")
st.markdown(
    "Click **Model** in the left sidebar to open the live allocation dashboard. "
    "Everything below refers to controls on that page."
)

st.markdown("### Step 2 — Configure the sidebar")

g1, g2 = st.columns(2)
with g1:
    with st.expander("Asset tickers", expanded=True):
        st.markdown(
            "Enter any two Yahoo Finance tickers. Examples:\n"
            "- **SPY / QQQ** — large-cap equity vs growth tech\n"
            "- **SPY / GLD** — equity vs gold (low correlation)\n"
            "- **QQQ / TLT** — growth tech vs long bonds (negative correlation in risk-off)\n"
            "- **BTC-USD / ETH-USD** — crypto pair\n\n"
            "Avoid very illiquid tickers — the price fetch may fail."
        )
    with st.expander("Risk aversion γ"):
        st.markdown(
            "Controls how aggressively the model invests.\n\n"
            "| γ | Behaviour |\n|---|---|\n"
            "| −0.1 | Near risk-neutral, near 100% equity when premium is positive |\n"
            "| −1.0 | Moderate, good starting point |\n"
            "| −3.0 | Conservative, large cash buffer |\n"
            "| −5.0 | Very conservative |\n\n"
            "If the model returns all-cash, try γ closer to 0 or check the "
            "**Why is the allocation what it is?** panel."
        )
    with st.expander("Basket weight α"):
        st.markdown(
            "Sets the terminal utility target: $U(\\alpha X + (1-\\alpha) Y)$.\n\n"
            "- **α = 0.5** — equal basket, symmetric treatment of both assets\n"
            "- **α = 0.8** — model optimises primarily for asset X performance\n"
            "- **α = 0.2** — model optimises primarily for asset Y performance\n\n"
            "This changes the shape of the policy surface — you can see the effect "
            "on the heatmaps after running."
        )
    with st.expander("Horizon T"):
        st.markdown(
            "The investment horizon in years. The model solves for the allocation that "
            "maximises expected utility at time T.\n\n"
            "- Short horizons (0.25–0.5 yr) → allocation closer to myopic Merton solution\n"
            "- Long horizons (2–3 yr) → more weight on mean-reversion and jump risk\n\n"
            "For active trading use 0.25–1 yr. For strategic allocation use 1–3 yr."
        )

with g2:
    with st.expander("Solver grid (N, J, M, n_pi)", expanded=True):
        st.markdown(
            "Controls the resolution and speed of the Bellman solver.\n\n"
            "| Setting | Recommended | Notes |\n|---|---|---|\n"
            "| N, J | 64 (fast) / 128 (accurate) | Grid points in x and y |\n"
            "| M | 20 | Time steps — 10 is fine for T ≤ 1yr |\n"
            "| n_pi | 11 | Policy candidates — 11 gives 66 policies |\n\n"
            "The memory estimate shown (🟢/🟡/🔴) guides your choice. "
            "Stay green for interactive use, yellow for overnight runs."
        )
    with st.expander("Account balance"):
        st.markdown(
            "Enter your portfolio value in dollars. Used only for display — "
            "it scales the dollar P&L, dollar allocation, and expected return metrics. "
            "It does not affect the model's allocation percentages."
        )
    with st.expander("KAN training epochs"):
        st.markdown(
            "After the Bellman solve, a **Residual KAN** (2→4→3, ~300 parameters) is trained "
            "to interpolate the policy surface using MSE loss.\n\n"
            "- **50–100 epochs**: sufficient when the policy surface is flat (one asset dominates)\n"
            "- **150 epochs**: good default — residual base converges fast, KAN refines edges\n"
            "- **300 epochs**: use when the policy surface has strong spatial gradients (e.g. QQQ/TLT)\n\n"
            "Much faster than the old MLP (400 epochs) due to the residual baseline absorbing "
            "the constant component immediately. Enables microsecond inference at live prices."
        )

st.markdown("### Step 3 — Run the pipeline")
st.markdown(
    "Press **▶ Run / Recalibrate**. The pipeline runs four steps in sequence:\n\n"
    "1. **Fetch data** — 2 years of daily closes from yfinance (free, 15-min delayed)\n"
    "2. **Calibrate** — fits σ, ρ, λ, μ̃, σ̃ to the return history; auto-selects drift shrinkage λ\n"
    "3. **Solve Bellman** — backward induction over M time steps on the N×J grid via FFT\n"
    "4. **Train KAN** — fits the Residual KAN (~300 params, MSE loss) on the grid solution\n\n"
    "At 64×64 this takes ~30 seconds. At 128×128 it takes ~3 minutes."
)

st.markdown("### Step 4 — Read the results")

r1, r2 = st.columns(2)
with r1:
    with st.expander("Current Positions — gauges and recommendation"):
        st.markdown(
            "The three gauges show the optimal allocation in percent. "
            "The bold line below summarises it: `X % · Y % · Cash %`. "
            "These update silently every 15 minutes as prices move — "
            "you do not need to re-run the full pipeline."
        )
    with st.expander("Portfolio P&L & Returns"):
        st.markdown(
            "- **Dollar Allocation** — your account balance split by the recommended percentages\n"
            "- **Expected Annual Returns** — annualised expected log-return for the optimal portfolio "
            "vs three benchmarks (50/50, all-X, all-cash)\n"
            "- **Hedging Profit** — the alpha the optimal portfolio is expected to generate vs 50/50\n"
            "- **P&L Since Last Calibration** — unrealised return since you last pressed Run, "
            "using the recommended allocation as if you had entered at those prices"
        )
    with st.expander("Optimal Policy Surface"):
        st.markdown(
            "Three heatmaps showing the optimal allocation across all possible price combinations:\n\n"
            "- **Blue (π_x)** — fraction in asset X at each (X price, Y price) point\n"
            "- **Green (π_y)** — fraction in asset Y\n"
            "- **Orange (Cash)** — cash fraction\n\n"
            "The red dot is the current price. A smooth gradient means the model is using "
            "the 2D price structure. A flat uniform colour means one asset dominates regardless of price."
        )
with r2:
    with st.expander("Calibrated Model Parameters"):
        st.markdown(
            "Shows what was estimated from the data:\n\n"
            "- **σ_x, σ_y** — annualised diffusion vols\n"
            "- **ρ** — Brownian correlation (negative = hedge relationship)\n"
            "- **λ** — jump intensity in events per year\n"
            "- **Jump days** — how many of the ~504 trading days contained a jump\n"
            "- **μ_x, μ_y blended** — the shrinkage-adjusted drift used by the solver, "
            "with the raw sample mean and CAPM prior shown as a delta\n"
            "- **auto λ** — the cross-validated shrinkage weight the model chose"
        )
    with st.expander("Why is the allocation what it is?"):
        st.markdown(
            "Diagnostic panel that checks whether each asset clears the CRRA hurdle:\n\n"
            "> Risky beats cash when `μ_real − r > r + 1.5σ²`\n\n"
            "- Green ✅ = equity premium clears the hurdle → model invests\n"
            "- Red ❌ = premium below hurdle → model prefers cash\n\n"
            "If you see ❌ but believe the asset is worth holding, either "
            "increase γ toward 0 (less risk aversion) or try a longer data lookback."
        )
    with st.expander("Options Overlay — Protective Puts"):
        st.markdown(
            "Prices Black-Scholes protective puts on your equity positions using the "
            "calibrated volatilities. Select expiry (1/3/6/12 months) and strike moneyness.\n\n"
            "Shows per-asset: put premium, total cost, breakeven return, delta, "
            "and 1-σ P&L comparison hedged vs unhedged. "
            "Also shows the implied vol smile across strikes."
        )
    with st.expander("Historical Backtest"):
        st.markdown(
            "Rolls the full pipeline (calibrate → solve) forward through 5 years of history, "
            "rebalancing at your chosen frequency. Uses N=64 for speed.\n\n"
            "- **Calibration window** — how many days of history each recalibration sees\n"
            "- **Rebalance frequency** — how often the portfolio is rebalanced\n"
            "- **γ for backtest** — can differ from the live γ\n\n"
            "The chart shows wealth growth vs three benchmarks. "
            "Dotted vertical lines mark recalibration dates. "
            "The performance table shows annualised return, vol, Sharpe, and max drawdown."
        )

st.markdown("---")

st.markdown("## Quick Reference")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Good starting settings**")
    st.markdown(
        "| Parameter | Value |\n|---|---|\n"
        "| Tickers | SPY / QQQ |\n"
        "| γ | −1.0 |\n"
        "| α | 0.5 |\n"
        "| Horizon | 1.0 yr |\n"
        "| N, J | 64 |\n"
        "| M | 20 |\n"
        "| n_pi | 11 |\n"
        "| KAN epochs | 150 |"
    )

with col_b:
    st.markdown("**Diagnostic checklist**")
    st.markdown(
        "If you get all-cash:\n"
        "1. Check the equity premium panel — is μ_real − r > hurdle?\n"
        "2. Try γ = −0.3 (less risk averse)\n"
        "3. Try a 5-year lookback (more data, stabler drift estimate)\n"
        "4. Check the risk-free rate — if r > 4%, the hurdle is harder to clear\n\n"
        "If the heatmap is flat:\n"
        "1. One asset likely dominates completely\n"
        "2. Try an asset pair with lower correlation (SPY/GLD, QQQ/TLT)"
    )

with col_c:
    st.markdown("**Data notes**")
    st.markdown(
        "- Prices are 15-minute delayed on the yfinance free tier\n"
        "- Live section auto-refreshes every 15 min without a full re-run\n"
        "- Risk-free rate is pulled from ^IRX (13-week T-bill)\n"
        "- The model uses 2 years of daily data for calibration (≈504 days)\n"
        "- The backtest pulls 5 years of data independently\n"
        "- Recalibrate whenever you change tickers, γ, or grid settings"
    )

st.markdown("---")
st.caption(
    "Model: Merton (1969) jump-diffusion · Solver: Zhou & Dang (2025) monotone integration · "
    "Utility: CRRA · Shrinkage: empirical Bayes via time-series CV · "
    "Policy net: Residual RBF-KAN (Liu et al., 2024)"
)
