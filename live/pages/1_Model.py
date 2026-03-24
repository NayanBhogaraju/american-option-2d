import os
import sys
import time
from typing import Optional

import numpy as np
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import plotly.graph_objects as go
from live.system import AllocationSystem
from live.options_overlay import options_overlay, vol_surface_smile

st.set_page_config(
    page_title="Live Allocation — Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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


def _init_state():
    if "system" not in st.session_state:
        st.session_state.system = None


def _sidebar() -> dict:
    st.sidebar.title("⚙️ Settings")

    ticker_x = st.sidebar.text_input("Asset X ticker", value="SPY").upper().strip()
    ticker_y = st.sidebar.text_input("Asset Y ticker", value="QQQ").upper().strip()

    st.sidebar.markdown("---")
    gamma = st.sidebar.slider(
        "Risk aversion γ (CRRA)",
        min_value=-5.0, max_value=-0.1, value=-1.0, step=0.1,
        help="γ < 0: risk averse. γ → -∞: extremely conservative.",
    )
    alpha = st.sidebar.slider(
        "Basket weight α",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Terminal utility U(αX + (1-α)Y). α=0.5 → equal basket.",
    )
    horizon = st.sidebar.slider(
        "Horizon T (years)", min_value=0.25, max_value=3.0, value=1.0, step=0.25,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Solver grid** (smaller = faster)")
    col1, col2 = st.sidebar.columns(2)
    N = col1.selectbox("N (x-grid)", [64, 128, 192], index=0)
    J = col2.selectbox("J (y-grid)", [64, 128, 192], index=0)
    M = st.sidebar.slider("M (time steps)", 10, 50, value=20, step=5)
    n_pi = st.sidebar.slider("n_pi (policy grid)", 5, 15, value=11, step=2)

    _tmp, _n_fft = 1, 1
    while _tmp < (2 * N + 1) + (3 * N - 1) - 1:
        _tmp <<= 1
        _n_fft = _tmp
    _n_pol = sum(
        1 for px in range(n_pi) for py in range(n_pi)
        if (px + py) / (n_pi - 1) <= 1.0 + 1e-6
    )
    _mem = _n_pol * _n_fft * (_n_fft // 2 + 1) * 16 / 1e6
    _dot = "🟢" if _mem < 300 else ("🟡" if _mem < 700 else "🔴")
    st.sidebar.caption(f"{_dot} ~{_mem:.0f} MB · {_n_pol} policies · FFT {_n_fft}²")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Account**")
    account_balance = st.sidebar.number_input(
        "Account balance ($)",
        min_value=1_000, max_value=100_000_000,
        value=100_000, step=1_000, format="%d",
        help="Total portfolio value — used for dollar P&L calculations.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Policy net: Residual KAN (2→4→3, RBF basis, MSE loss)")
    net_epochs = st.sidebar.slider("KAN training epochs", 50, 300, value=150, step=25)

    return dict(
        ticker_x=ticker_x, ticker_y=ticker_y,
        gamma=gamma, alpha=alpha, horizon=horizon,
        N=N, J=J, M=M, n_pi=n_pi,
        net_epochs=net_epochs,
        account_balance=account_balance,
    )


PIPELINE_STEPS = [
    "Fetching market data...",
    "Calibrating Merton model (auto drift shrinkage)...",
    "Solving Bellman equation...",
    "Training KAN policy network...",
]


def _run_pipeline_with_progress(cfg: dict) -> None:
    st.markdown("### Running pipeline")
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_box = st.empty()
    log_lines = []
    n_steps = len(PIPELINE_STEPS)
    status_text.info(f"Step 0/{n_steps} — Initialising...")

    try:
        sys_obj = AllocationSystem(
            ticker_x=cfg["ticker_x"],
            ticker_y=cfg["ticker_y"],
            gamma=cfg["gamma"],
            alpha=cfg["alpha"],
            horizon_years=cfg["horizon"],
            grid_kwargs=dict(N=cfg["N"], J=cfg["J"], M=cfg["M"], n_pi=cfg["n_pi"]),
            net_epochs=cfg["net_epochs"],
        )
        for completed, msg in sys_obj.run_pipeline_steps(verbose=False):
            pct = int(completed / n_steps * 100)
            progress_bar.progress(pct)
            next_label = PIPELINE_STEPS[completed] if completed < n_steps else "Finishing..."
            status_text.info(f"Step {completed}/{n_steps} done — next: {next_label}")
            log_lines.append(f"✓ {msg}")
            log_box.code("\n".join(log_lines))

        progress_bar.progress(100)
        status_text.success(
            f"Pipeline complete!  N={cfg['N']} J={cfg['J']} M={cfg['M']} γ={cfg['gamma']}"
        )
        st.session_state.system = sys_obj
        st.session_state.pipeline_log = "\n".join(log_lines)

    except Exception as e:
        import traceback
        progress_bar.progress(0)
        status_text.error("Pipeline failed — see log above")
        log_lines.append(f"\n❌ ERROR: {e}")
        log_box.code("\n".join(log_lines) + "\n\n" + traceback.format_exc())
        st.session_state.pipeline_error = f"{e}\n\n{traceback.format_exc()}"
        st.session_state.pipeline_log = "\n".join(log_lines)


def _gauge(label: str, value: float, color: str = "#2196F3") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 30, "color": "white"}},
        title={"text": label, "font": {"size": 12, "color": "rgba(255,255,255,0.6)"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.2)", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.6},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps": [{"range": [0, 100], "color": "rgba(255,255,255,0.05)"}],
        },
    ))
    fig.update_layout(
        height=170,
        margin=dict(t=20, b=0, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _heatmap(z, x, y, title, colorscale, zmin=0, zmax=1) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.T,
        x=np.round(np.exp(x), 2),
        y=np.round(np.exp(y), 2),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar={"title": title, "thickness": 10, "titlefont": {"size": 11}},
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="rgba(255,255,255,0.85)")),
        xaxis_title="Asset X price",
        yaxis_title="Asset Y price",
        height=300,
        margin=dict(t=35, b=40, l=50, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.fragment(run_every=900)
def _live_section(system: AllocationSystem, cfg: dict):
    try:
        status = system.status()
    except Exception as e:
        st.warning(f"Price fetch failed: {e}")
        return

    bal     = cfg["account_balance"]
    pi_x    = status.get("pi_x")    or 0.0
    pi_y    = status.get("pi_y")    or 0.0
    pi_cash = status.get("pi_cash") or 0.0

    # ── Allocation card ─────────────────────────────────────────────────────
    with st.container(border=True):
        # Prices + gauges in one row
        p1, p2, g1, g2, g3 = st.columns([1, 1, 1.2, 1.2, 1.2])
        p1.metric(cfg["ticker_x"], f"${status.get('X_price', 0):.2f}")
        p2.metric(cfg["ticker_y"], f"${status.get('Y_price', 0):.2f}")
        with g1:
            st.plotly_chart(_gauge(f"{cfg['ticker_x']}", pi_x, "#2196F3"), use_container_width=True)
        with g2:
            st.plotly_chart(_gauge(f"{cfg['ticker_y']}", pi_y, "#4CAF50"), use_container_width=True)
        with g3:
            st.plotly_chart(_gauge("Cash", pi_cash, "#FF9800"), use_container_width=True)

        st.markdown(
            f"**Recommended allocation:** "
            f"{cfg['ticker_x']} **{pi_x*100:.1f}%** &nbsp;·&nbsp; "
            f"{cfg['ticker_y']} **{pi_y*100:.1f}%** &nbsp;·&nbsp; "
            f"Cash **{pi_cash*100:.1f}%**",
            unsafe_allow_html=True,
        )

        # Dollar allocation
        st.markdown("**Dollar Allocation**")
        da1, da2, da3, da4 = st.columns(4)
        da1.metric("Total balance",     f"${bal:,.0f}")
        da2.metric(cfg["ticker_x"],     f"${bal*pi_x:,.0f}",    delta=f"{pi_x*100:.1f}%")
        da3.metric(cfg["ticker_y"],     f"${bal*pi_y:,.0f}",    delta=f"{pi_y*100:.1f}%")
        da4.metric("Cash / Bond",       f"${bal*pi_cash:,.0f}", delta=f"{pi_cash*100:.1f}%")

    # ── Returns & P&L (collapsible) ─────────────────────────────────────────
    er_opt  = status.get("expected_return_optimal")
    er_5050 = status.get("expected_return_50_50")
    er_allx = status.get("expected_return_all_x")
    er_cash = status.get("expected_return_cash")

    with st.expander("Expected Returns & P&L", expanded=True):
        if er_opt is not None:
            st.markdown("**Expected Annual Returns vs Benchmarks**")
            benchmarks = {
                "Optimal (model)":        er_opt,
                "50/50 fixed":            er_5050,
                f"All {cfg['ticker_x']}": er_allx,
                "All cash":               er_cash,
            }
            for col, (label, er) in zip(st.columns(4), benchmarks.items()):
                dvc = (er - er_cash) if er is not None and er_cash is not None else None
                col.metric(
                    label,
                    f"${bal*er:,.0f}/yr" if er is not None else "—",
                    delta=f"{dvc*100:+.2f}% vs cash" if dvc is not None else None,
                    delta_color="normal",
                )

            if er_5050 is not None:
                hp = er_opt - er_5050
                st.markdown("**Alpha vs 50/50**")
                hp1, hp2, hp3 = st.columns(3)
                hp1.metric("Annual alpha",    f"{hp*100:+.2f}%",
                           delta=f"${bal*hp:+,.0f}/yr",
                           delta_color="normal" if hp >= 0 else "inverse")
                hp2.metric("Optimal return",  f"{er_opt*100:.2f}%",
                           delta=f"${bal*er_opt:,.0f}/yr", delta_color="normal")
                hp3.metric("50/50 return",    f"{er_5050*100:.2f}%",
                           delta=f"${bal*er_5050:,.0f}/yr", delta_color="normal")

        x_entry = status.get("X_entry")
        y_entry = status.get("Y_entry")
        x_now   = status.get("X_price")
        y_now   = status.get("Y_price")

        if all(v is not None and v > 0 for v in [x_entry, y_entry, x_now, y_now]):
            ret_x = (x_now - x_entry) / x_entry
            ret_y = (y_now - y_entry) / y_entry
            pnl_x = bal * pi_x * ret_x
            pnl_y = bal * pi_y * ret_y
            pnl_total  = pnl_x + pnl_y
            bench_pnl  = bal * 0.5 * ret_x + bal * 0.5 * ret_y

            st.markdown("**P&L Since Last Calibration**")
            pl1, pl2, pl3, pl4 = st.columns(4)
            pl1.metric(f"{cfg['ticker_x']} return", f"{ret_x*100:+.2f}%",
                       delta=f"${pnl_x:+,.0f}",
                       delta_color="normal" if pnl_x >= 0 else "inverse")
            pl2.metric(f"{cfg['ticker_y']} return", f"{ret_y*100:+.2f}%",
                       delta=f"${pnl_y:+,.0f}",
                       delta_color="normal" if pnl_y >= 0 else "inverse")
            pl3.metric("Total P&L", f"${pnl_total:+,.0f}",
                       delta=f"{pnl_total/bal*100:+.2f}%",
                       delta_color="normal" if pnl_total >= 0 else "inverse")
            pl4.metric("vs 50/50", f"${pnl_total - bench_pnl:+,.0f}",
                       delta=f"bench ${bench_pnl:+,.0f}",
                       delta_color="normal" if (pnl_total - bench_pnl) >= 0 else "inverse")

            cal_ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(status.get("last_pipeline_time", 0)))
            st.caption(
                f"Entry at calibration ({cal_ts}): "
                f"{cfg['ticker_x']} ${x_entry:.2f} · {cfg['ticker_y']} ${y_entry:.2f}"
            )


def _options_section(system: AllocationSystem, cfg: dict, status: dict):
    st.markdown("### Options Overlay — Protective Puts")
    st.caption(
        "Black-Scholes benchmark price shown alongside the **utility-indifference price** "
        "from a second Bellman solve with put-modified terminal utility. "
        "The indifference price is what a CRRA investor would pay to eliminate the downside risk."
    )

    sigma_x = status.get("sigma_x") or 0.15
    sigma_y = status.get("sigma_y") or 0.18
    r = status.get("r") or 0.05
    pi_x = status.get("pi_x") or 0.0
    pi_y = status.get("pi_y") or 0.0
    S_x = status.get("X_price") or 1.0
    S_y = status.get("Y_price") or 1.0
    bal = cfg["account_balance"]

    oc1, oc2, oc3 = st.columns(3)
    opt_T = oc1.selectbox("Expiry (months)", [1, 3, 6, 12], index=1, key="opt_T") / 12
    moneyness = oc2.slider("Strike moneyness", 0.80, 1.00, value=0.95, step=0.01, key="opt_K")
    oc3.caption(
        f"ATM vols: σ_x={sigma_x*100:.1f}% σ_y={sigma_y*100:.1f}%\n"
        f"Risk-free: {r*100:.2f}%"
    )

    result = options_overlay(
        cfg["ticker_x"], cfg["ticker_y"],
        S_x, S_y, sigma_x, sigma_y, r,
        pi_x, pi_y, bal, T=opt_T, moneyness=moneyness,
        system=system,
    )

    quotes = result["quotes"]
    if not quotes:
        st.info("No equity allocation — nothing to hedge.")
        return

    qcols = st.columns(len(quotes))
    for col, (ticker, q) in zip(qcols, quotes.items()):
        col.markdown(f"**{ticker} put  (K={q.K:.2f}, T={opt_T*12:.0f}mo)**")
        col.metric("Put premium / share", f"${q.put_price_per_share:.2f}",
                   delta=f"{q.put_price_pct*100:.2f}% of spot")
        col.metric("Position value", f"${q.position_value:,.0f}")
        col.metric("Total put cost", f"${q.position_value * q.put_price_pct:,.0f}",
                   delta=f"{q.put_price_pct*100:.2f}%")
        col.metric("Breakeven return", f"{q.breakeven_return*100:.2f}%")
        col.metric("Delta", f"{q.delta:.3f}")
        col.caption(
            f"1-σ unhedged: ${q.unhedged_downside_1sd:,.0f}  |  "
            f"1-σ hedged: ${q.hedged_downside_1sd:,.0f}"
        )

    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("BS total hedge cost", f"${result['total_hedge_cost_dollars']:,.0f}",
               delta=f"{result['total_hedge_cost_pct']*100:.2f}% of portfolio")
    tc2.metric("Equity exposure", f"${result['total_equity_value']:,.0f}")
    tc3.metric("Hedged portfolio value",
               f"${bal - result['total_hedge_cost_dollars']:,.0f}")

    indif_pct = result.get("indif_price_pct")
    indif_usd = result.get("indif_price_dollars")
    if indif_pct is not None:
        bs_pct = result["total_hedge_cost_pct"]
        delta_str = f"{(indif_pct - bs_pct)*100:+.2f}% vs BS"
        tc4.metric(
            "Indifference price",
            f"${indif_usd:,.0f}",
            delta=f"{indif_pct*100:.2f}% of portfolio  ({delta_str})",
            delta_color="off",
            help=(
                "Utility-indifference price from a second Bellman solve with "
                "put-modified terminal utility. Uses fast grid N=64, M=5."
            ),
        )
    else:
        with tc4:
            with st.spinner("Computing indifference price..."):
                pass
            st.caption("Indifference price: run pipeline first")

    with st.expander("Implied vol smile"):
        fig_smile = go.Figure()
        for ticker, sigma, S in [
            (cfg["ticker_x"], sigma_x, S_x),
            (cfg["ticker_y"], sigma_y, S_y),
        ]:
            strikes, vols, puts = vol_surface_smile(S, r, opt_T, sigma)
            fig_smile.add_trace(go.Scatter(
                x=strikes / S * 100, y=vols * 100,
                mode="lines+markers", name=f"{ticker} implied vol",
            ))
        fig_smile.update_layout(
            xaxis_title="Moneyness (% of spot)",
            yaxis_title="Implied vol (%)",
            height=300, margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig_smile, use_container_width=True)


def main():
    _init_state()
    cfg = _sidebar()

    st.markdown(f"## 📈 {cfg['ticker_x']} / {cfg['ticker_y']} — Live Allocation")

    # ── Run button row ──────────────────────────────────────────────────────
    btn_col, status_col = st.columns([1, 3])
    run_clicked = btn_col.button("▶ Run / Recalibrate", type="primary", use_container_width=True)

    system: Optional[AllocationSystem] = st.session_state.system
    if system is not None:
        try:
            s = system.status()
            last = s.get("last_pipeline_time", 0)
            dur  = s.get("pipeline_duration_s", 0)
            ts   = time.strftime("%H:%M:%S", time.localtime(last)) if last else "—"
            status_col.caption(
                f"Last run: **{ts}** · {dur:.1f}s · "
                f"γ={cfg['gamma']} · N={cfg['N']}×{cfg['J']} · "
                f"KAN {'✓' if s.get('net_ready') else '…'} · "
                f"prices ~15 min delayed"
            )
        except Exception:
            pass

    if run_clicked:
        _run_pipeline_with_progress(cfg)
        if "pipeline_error" not in st.session_state:
            st.rerun()

    if "pipeline_log" in st.session_state:
        with st.expander("Pipeline log", expanded=False):
            st.code(st.session_state.pipeline_log)

    if "pipeline_error" in st.session_state:
        with st.expander("⚠️ Pipeline error", expanded=True):
            st.code(st.session_state.pipeline_error)
        if st.button("Clear error"):
            del st.session_state["pipeline_error"]
            st.rerun()

    if system is None:
        st.info("Configure the sidebar and press **▶ Run / Recalibrate** to start.")
        st.stop()

    # ── Live allocation (auto-refreshes every 15 min) ──────────────────────
    _live_section(system, cfg)

    # ── Policy surface ─────────────────────────────────────────────────────
    surface = system.policy_surface()
    if surface is not None:
        st.markdown("### Optimal Policy Surface")
        try:
            status_now = system.status()
            px_s = status_now.get("X_price")
            py_s = status_now.get("Y_price")
        except Exception:
            px_s = py_s = None

        if system._net is not None:
            pi_x_surf, pi_y_surf = system._net.predict_grid(surface["x"], surface["y"])
        else:
            pi_x_surf = surface["pi_x"]
            pi_y_surf = surface["pi_y"]
        pi_cash_surf = np.clip(1.0 - pi_x_surf - pi_y_surf, 0, 1)

        hc1, hc2, hc3 = st.columns(3)
        for ax, z, title, cs in [
            (hc1, pi_x_surf,    f"π_x  ({cfg['ticker_x']})", "Blues"),
            (hc2, pi_y_surf,    f"π_y  ({cfg['ticker_y']})", "Greens"),
            (hc3, pi_cash_surf, "Cash fraction",              "Oranges"),
        ]:
            with ax:
                fig = _heatmap(z, surface["x"], surface["y"], title, cs)
                if px_s and py_s:
                    fig.add_scatter(
                        x=[px_s], y=[py_s], mode="markers",
                        marker=dict(color="red", size=9, symbol="x"), name="Now",
                    )
                st.plotly_chart(fig, use_container_width=True)

    # ── Analysis tabs ───────────────────────────────────────────────────────
    static_status = system.status()

    tab_cal, tab_ep, tab_opt = st.tabs(["📐 Calibration", "⚖️ Equity Premium", "🛡️ Options Overlay"])

    with tab_cal:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**Diffusion**")
            sx = static_status.get("sigma_x")
            sy = static_status.get("sigma_y")
            rho = static_status.get("rho")
            st.metric(f"σ_x  ({cfg['ticker_x']})", f"{sx*100:.1f}%" if sx else "—")
            st.metric(f"σ_y  ({cfg['ticker_y']})", f"{sy*100:.1f}%" if sy else "—")
            st.metric("ρ  (Brownian corr)", f"{rho:.3f}" if rho is not None else "—")
        with mc2:
            st.markdown("**Jumps**")
            lam = static_status.get("lam")
            nj  = static_status.get("n_jump_days")
            nt  = static_status.get("n_total_days")
            st.metric("λ  (jumps / yr)", f"{lam:.2f}" if lam is not None else "—")
            if nj is not None and nt:
                st.metric("Jump days", f"{nj} / {nt}  ({100*nj/nt:.1f}%)")
        with mc3:
            st.markdown("**Real-world drifts**")
            mux     = static_status.get("mu_x_real")
            muy     = static_status.get("mu_y_real")
            mux_raw = static_status.get("mu_x_raw")
            muy_raw = static_status.get("mu_y_raw")
            mux_pri = static_status.get("mu_x_prior")
            muy_pri = static_status.get("mu_y_prior")
            bx      = static_status.get("beta_x")
            by      = static_status.get("beta_y")
            shrink  = static_status.get("shrinkage") or 0.0
            st.metric(
                f"μ_x  ({cfg['ticker_x']}) blended",
                f"{mux*100:.1f}%" if mux is not None else "—",
                delta=f"raw {mux_raw*100:.1f}% → prior {mux_pri*100:.1f}%" if mux_raw is not None else None,
                delta_color="off",
            )
            st.metric(
                f"μ_y  ({cfg['ticker_y']}) blended",
                f"{muy*100:.1f}%" if muy is not None else "—",
                delta=f"raw {muy_raw*100:.1f}% → prior {muy_pri*100:.1f}%" if muy_raw is not None else None,
                delta_color="off",
            )
            if bx is not None:
                st.caption(f"β_x={bx:.2f}  β_y={by:.2f}  auto shrinkage λ={shrink:.2f}")

        st.markdown("---")
        net_ready = static_status.get("net_ready", False)
        if net_ready:
            st.success("KAN ready — 2→4→3 RBF-KAN · residual baseline · MSE loss · ~300 params")
        else:
            st.info("KAN not yet trained — run pipeline first")
        st.caption(f"γ={cfg['gamma']} · α={cfg['alpha']} · T={cfg['horizon']} yr · N={cfg['N']}×{cfg['J']} · M={cfg['M']} · n_pi={cfg['n_pi']}")

    with tab_ep:
        ep_x     = static_status.get("equity_premium_x")
        ep_y     = static_status.get("equity_premium_y")
        hurdle_x = static_status.get("hurdle_x")
        hurdle_y = static_status.get("hurdle_y")

        if ep_x is None:
            st.info("Run the pipeline to see equity premium diagnostics.")
        else:
            r_actual = None
            try:
                r_actual = system._feed.risk_free_rate()
            except Exception:
                pass

            st.markdown(
                "An asset is preferred over cash when its equity premium `μ_real − r` "
                "exceeds the CRRA hurdle `r + 1.5σ²`."
            )
            dc1, dc2 = st.columns(2)
            for col, label, ep, hurdle in [
                (dc1, cfg["ticker_x"], ep_x, hurdle_x),
                (dc2, cfg["ticker_y"], ep_y, hurdle_y),
            ]:
                if ep is not None and hurdle is not None:
                    beats = ep > (hurdle - (r_actual or 0.05))
                    icon = "✅" if beats else "❌"
                    col.metric(
                        f"{icon}  {label} equity premium",
                        f"{ep*100:+.1f}%",
                        delta=f"hurdle {hurdle*100:.1f}%",
                        delta_color="normal" if beats else "inverse",
                    )
                    col.caption("Risky preferred over cash" if beats else "Cash preferred — try γ closer to 0")
            if r_actual:
                st.caption(f"Risk-free rate (^IRX): {r_actual*100:.2f}%")

    with tab_opt:
        _options_section(system, cfg, static_status)

    st.caption(
        "📊 Historical backtest & stats → **Backtest** page in the sidebar  ·  "
        "Prices: 15 min delayed (yfinance free tier)  ·  auto-refresh every 15 min"
    )


main()
