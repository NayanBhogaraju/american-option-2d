import streamlit as st

st.set_page_config(
    page_title="Two-Asset Allocation System",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem !important; padding-bottom: 2rem; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border-color: rgba(255,255,255,0.1) !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.title("Two-Asset Optimal Allocation System")
st.caption("Merton Jump-Diffusion · Bellman Equation · CRRA Utility · Surrogate KAN · Live Market Data")

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
        "Fetches market data, fits a Merton jump-diffusion model, solves a "
        "Bellman equation on a 128×128 price grid via FFT convolution, "
        "trains a **Residual KAN** on the solution for live inference, "
        "and distils 20 years of Bellman solutions into a **parameter-conditioned "
        "Surrogate KAN** for fast historical backtesting — including "
        "bootstrap confidence intervals and utility-indifference options pricing."
    )
with c3:
    st.markdown("#### What makes it different")
    st.markdown(
        "Most allocators use Markowitz (static, one-shot). This one "
        "is **dynamic** — it accounts for how assets jump, how they "
        "co-move, and how the optimal allocation changes as prices "
        "evolve. The surrogate KAN compresses thousands of Bellman "
        "solves into a single forward pass, making 20-year rolling "
        "backtests tractable in seconds."
    )

st.markdown("---")


# ── Math ──────────────────────────────────────────────────────────────────────

st.markdown("## How It Works — The Math")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Asset Model", "Bellman Equation", "Green's Function",
    "CRRA Utility", "Drift Shrinkage", "Live KAN",
    "Surrogate KAN", "Indifference Pricing",
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
    st.markdown("### Live Residual KAN Policy Network")
    st.markdown(
        "After the Bellman solve, a compact neural network is trained on the resulting "
        "policy grid so that inference at live prices is instantaneous, without re-running the solver. "
        "This is the **live KAN** — it is specific to one set of calibration parameters and prices."
    )

    st.markdown("#### Architecture — 2 → 4 → 3 RBF-KAN (~300 parameters)")
    st.markdown(
        "Two components add together before the final softmax:"
    )
    st.latex(r"""
        \pi(x, y) \;=\; \text{softmax}\!\Bigl(\underbrace{\mathbf{b}}_{\text{residual base}} \;+\; \underbrace{\text{KAN}(x,\, y)}_{\text{spatial correction}}\Bigr)
    """)
    st.markdown(
        "**Residual base** $\\mathbf{b} \\in \\mathbb{R}^3$ quickly converges to the dominant "
        "allocation (e.g. [0.7, 0.0, 0.3]) in the first few epochs, "
        "so the KAN only needs to learn the small spatial deviation from that baseline."
    )

    st.markdown("#### KAN layers — RBF basis functions")
    st.markdown(
        "Each **edge** $(i \\to j)$ has its own learned univariate function $\\varphi_{ij}$:"
    )
    st.latex(r"""
        \text{KAN layer output}_j \;=\; \sum_{i} \varphi_{ij}(x_i)
        \;+\; \sum_{i} w_{ij}^{\text{base}} \, x_i
    """)
    st.markdown(
        "Each $\\varphi_{ij}$ is parameterised as a weighted sum of Gaussian RBF basis functions "
        "centred on a fixed grid of 12 points over $[-4, 4]$:"
    )
    st.latex(r"""
        \varphi_{ij}(x) \;=\; \sum_{k=1}^{12} c_{ijk} \exp\!\left(-\frac{(x - \mu_k)^2}{2h^2}\right)
    """)

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Parameters", "~300", delta="-4,247 vs MLP", delta_color="normal")
    kc2.metric("Training epochs", "150", delta="-250 vs MLP", delta_color="normal")
    kc3.metric("Inference time", "<1 ms", delta_color="off")

with tab7:
    st.markdown("### Surrogate KAN — Amortized Bellman for Backtesting")
    st.markdown(
        "The live KAN is trained on **one** Bellman solution at the current calibration point. "
        "For a 20-year rolling backtest with monthly rebalancing, you would need ~240 separate "
        "Bellman solves — roughly 2 hours of compute. "
        "The **surrogate KAN** solves this by learning the Bellman solution as a function of "
        "the calibration parameters themselves, not just the prices."
    )

    st.markdown("#### Architecture — 9 → 24 → 12 → 3 RBF-KAN (~7,000 parameters)")
    st.markdown(
        "Input is the full 9-dimensional parameter vector plus price displacements, "
        "normalised to $[-1, 1]$:"
    )
    st.latex(r"""
        \text{input} \;=\; (\sigma_x,\, \sigma_y,\, \rho,\, \lambda,\, \mu_x,\, \mu_y,\, r,\, \Delta x,\, \Delta y)
    """)
    st.latex(r"""
        (\pi_x,\, \pi_y,\, \pi_c) \;=\; \text{softmax}\!\bigl(\mathbf{b} + \text{KAN}_3(\text{SiLU}(\text{KAN}_2(\text{SiLU}(\text{KAN}_1(\text{input})))))\bigr)
    """)

    st.markdown("#### Training")
    st.markdown(
        "200 Bellman solutions are pre-computed offline on a fast grid (N=32, M=5) "
        "spanning the full parameter space — including extreme regimes "
        "like TLT's 2022 crash (μ ≈ −0.30) and crypto-level volatility. "
        "The training set contains ~200,000 (parameter, price, allocation) triples."
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.markdown(
        "**Parameter ranges trained on**\n"
        "- σ: 5% – 80%\n"
        "- ρ: −0.9 – +0.9\n"
        "- λ: 0 – 25 jumps/yr\n"
        "- μ: −35% – +50%\n"
        "- r: 0.5% – 10%\n"
    )
    sc2.markdown(
        "**Build once, use always**\n\n"
        "The surrogate is built automatically the first time you click "
        "**▶ Run Backtest** (~8 min). It is then cached on disk. "
        "If you change γ, α, or T, it auto-rebuilds for the new settings."
    )
    sc3.markdown(
        "**Validation**\n\n"
        "After loading, the surrogate is validated against one exact Bellman solve "
        "at a reference parameter point. If the max allocation error exceeds **5 pp**, "
        "an error is raised and the surrogate is rebuilt."
    )

    st.markdown("#### Speed comparison")
    speed_data = {
        "Method": ["Exact Bellman (N=64, M=10)", "Surrogate KAN forward pass"],
        "Time per rebalance": ["~30 s", "~1 ms"],
        "20-year backtest (240 rebalances)": ["~2 hours", "~1 second"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(speed_data).set_index("Method"), use_container_width=True)

with tab8:
    st.markdown("### Utility-Indifference Options Pricing")
    st.markdown(
        "Standard Black-Scholes prices options assuming the underlying can be continuously hedged. "
        "For an investor with a fixed allocation and CRRA preferences, "
        "the relevant price is the **utility-indifference price**: "
        "the maximum premium they would pay to floor the X component of their basket."
    )

    st.markdown("#### Modified terminal utility with put")
    st.markdown(
        "A protective put with normalised strike $k = K / S_{x,0}$ floors the X payoff, "
        "giving a modified terminal utility:"
    )
    st.latex(r"""
        U_{\text{put}}(x, y) \;=\; \frac{\bigl(\alpha \cdot \max(e^x,\, k) + (1-\alpha)\cdot e^y\bigr)^\gamma}{\gamma}
    """)

    st.markdown("#### Indifference price")
    st.markdown(
        "By CRRA scale invariance, the indifference price as a fraction of portfolio value is:"
    )
    st.latex(r"""
        p^* \;=\; W_0 \cdot \left(1 - \left(\frac{\bar{V}_{\text{no put}}}{\bar{V}_{\text{with put}}}\right)^{\!1/\gamma}\right)
    """)
    st.markdown(
        "where $\\bar{V}$ is the normalised value at the initial state from each Bellman solve. "
        "Both solves use **the same fast grid** (N=64, M=5) so the ratio is dimensionally consistent. "
        "The Black-Scholes price is shown alongside as a benchmark."
    )
    st.markdown(
        "**Interpretation**: $p^* > p_{BS}$ means the investor values the put more than the market "
        "charges — because their concentrated CRRA portfolio benefits more from the downside floor "
        "than the BS delta-hedger assumes."
    )

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
            "For active trading use 0.25–1 yr. For strategic allocation use 1–3 yr.\n\n"
            "**Note:** changing T invalidates the surrogate cache — it will auto-rebuild "
            "the next time you run a backtest."
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
            "After the Bellman solve, a **live Residual KAN** (2→4→3, ~300 parameters) is trained "
            "to interpolate the policy surface for microsecond inference at live prices.\n\n"
            "- **50–100 epochs**: sufficient when the policy surface is flat (one asset dominates)\n"
            "- **150 epochs**: good default — residual base converges fast, KAN refines edges\n"
            "- **300 epochs**: use when the policy surface has strong spatial gradients (e.g. QQQ/TLT)\n\n"
            "This is separate from the **Surrogate KAN** used in backtesting, which is built once "
            "and cached on disk."
        )

st.markdown("### Step 3 — Run the pipeline")
st.markdown(
    "Press **▶ Run / Recalibrate**. The pipeline runs four steps in sequence:\n\n"
    "1. **Fetch data** — 2 years of daily closes from yfinance (free, 15-min delayed)\n"
    "2. **Calibrate** — fits σ, ρ, λ, μ̃, σ̃ to the return history; auto-selects drift shrinkage λ\n"
    "3. **Solve Bellman** — backward induction over M time steps on the N×J grid via FFT\n"
    "4. **Train live KAN** — fits the Residual KAN (~300 params, MSE loss) on the grid solution\n\n"
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
    with st.expander("Options Overlay — Protective Puts & Indifference Pricing"):
        st.markdown(
            "Prices protective puts on your equity positions using two methods:\n\n"
            "**Black-Scholes price** — standard BSM formula using calibrated volatility. "
            "Select expiry (1/3/6/12 months) and strike moneyness. "
            "Shows per-asset: put premium, total cost, breakeven return, delta, "
            "and 1-σ P&L comparison hedged vs unhedged. Also shows the implied vol smile.\n\n"
            "**Utility-indifference price** — derived from a second Bellman solve with a "
            "put-modified terminal utility. This is the *maximum* premium a CRRA investor "
            "would rationally pay for the floor, given their risk aversion γ. "
            "When the indifference price exceeds the BS price, the put is cheap relative "
            "to the investor's personal valuation."
        )

st.markdown("### Step 5 — Run the Backtest")
st.markdown(
    "Click **Backtest** in the left sidebar to open the historical backtest page.\n\n"
    "**First run:** clicking **▶ Run Backtest** automatically builds a **Surrogate KAN** "
    "(200 Bellman solves, ~8 min) before starting the backtest. This is a one-time cost — "
    "the surrogate is cached and reused for all future backtests with the same γ / α / T.\n\n"
    "**Changing settings:** if γ, α, or T change between runs, the surrogate is automatically "
    "rebuilt for the new settings. No manual steps required.\n\n"
    "After the backtest completes, press **🔁 Run Bootstrap CI** to compute 95% confidence "
    "intervals on Sharpe ratio, annual return, and max drawdown via stationary block bootstrap "
    "(500 iterations, 21-day blocks). The significance test reports the bootstrapped p-value "
    "for H₀: Sharpe(optimal) ≤ Sharpe(50/50)."
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
        "| KAN epochs | 150 |\n"
        "| Backtest lookback | 10–20 yr |"
    )

with col_b:
    st.markdown("**Diagnostic checklist**")
    st.markdown(
        "If you get all-cash:\n"
        "1. Check the equity premium panel — is μ_real − r > hurdle?\n"
        "2. Try γ = −0.3 (less risk averse)\n"
        "3. Try a 5-year lookback (more data, stabler drift estimate)\n"
        "4. Check the risk-free rate — if r > 4%, the hurdle is harder to clear\n\n"
        "If the backtest vol looks too low (~cash-level):\n"
        "1. The surrogate may have been trained on a narrow parameter range\n"
        "2. It will auto-rebuild with the corrected ranges on next run\n\n"
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
        "- Backtest supports up to 20-year lookback (data availability varies by ticker)\n"
        "- Surrogate KAN is cached in `live/surrogate_kan.pt` and auto-rebuilds on param change\n"
        "- Recalibrate whenever you change tickers, γ, or grid settings"
    )

st.markdown("---")
st.caption(
    "Model: Merton (1969) jump-diffusion · Solver: Zhou & Dang (2025) monotone integration · "
    "Utility: CRRA · Shrinkage: empirical Bayes via time-series CV · "
    "Live policy net: Residual RBF-KAN 2→4→3 (Liu et al., 2024) · "
    "Surrogate: parameter-conditioned RBF-KAN 9→24→12→3 · "
    "Indifference pricing: Hodges & Neuberger (1989)"
)
