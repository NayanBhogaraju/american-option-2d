import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import plotly.graph_objects as go
from live.system import AllocationSystem
from live.backtest import run_backtest, BacktestResult
from live.options_overlay import options_overlay, vol_surface_smile
from live.data_feed import DataFeed

st.set_page_config(
    page_title="Live Allocation — Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def _gauge(label: str, value: float, color: str = "#1f77b4") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": label, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33],  "color": "#f0f2f6"},
                {"range": [33, 66], "color": "#e8ecf4"},
                {"range": [66, 100], "color": "#dde4f0"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=0, l=20, r=20))
    return fig


def _heatmap(z, x, y, title, colorscale, zmin=0, zmax=1) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.T,
        x=np.round(np.exp(x), 2),
        y=np.round(np.exp(y), 2),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar={"title": title, "thickness": 12},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Asset X price",
        yaxis_title="Asset Y price",
        height=320,
        margin=dict(t=40, b=40, l=60, r=20),
    )
    return fig


@st.fragment(run_every=900)
def _live_section(system: AllocationSystem, cfg: dict):
    try:
        status = system.status()
    except Exception as e:
        st.warning(f"Price fetch failed: {e}")
        return

    bal = cfg["account_balance"]
    pi_x    = status.get("pi_x")    or 0.0
    pi_y    = status.get("pi_y")    or 0.0
    pi_cash = status.get("pi_cash") or 0.0

    st.markdown("### Current Positions")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(cfg["ticker_x"], f"${status.get('X_price', 0):.2f}")
    c2.metric(cfg["ticker_y"], f"${status.get('Y_price', 0):.2f}")
    with c3:
        st.plotly_chart(_gauge(f"π_x ({cfg['ticker_x']})", pi_x, "#2196F3"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(_gauge(f"π_y ({cfg['ticker_y']})", pi_y, "#4CAF50"),
                        use_container_width=True)
    with c5:
        st.plotly_chart(_gauge("Cash", pi_cash, "#FF9800"),
                        use_container_width=True)

    st.markdown(
        f"**Recommended:** {cfg['ticker_x']} **{pi_x*100:.1f}%** · "
        f"{cfg['ticker_y']} **{pi_y*100:.1f}%** · Cash **{pi_cash*100:.1f}%**"
    )

    st.divider()

    st.markdown("### Portfolio P&L & Returns")

    st.markdown("#### Dollar Allocation")
    da1, da2, da3, da4 = st.columns(4)
    da1.metric("Total balance", f"${bal:,.0f}")
    da2.metric(cfg["ticker_x"],  f"${bal*pi_x:,.0f}", delta=f"{pi_x*100:.1f}%")
    da3.metric(cfg["ticker_y"],  f"${bal*pi_y:,.0f}", delta=f"{pi_y*100:.1f}%")
    da4.metric("Cash / Bond",    f"${bal*pi_cash:,.0f}", delta=f"{pi_cash*100:.1f}%")

    er_opt  = status.get("expected_return_optimal")
    er_5050 = status.get("expected_return_50_50")
    er_allx = status.get("expected_return_all_x")
    er_cash = status.get("expected_return_cash")

    if er_opt is not None:
        st.markdown("#### Expected Annual Returns vs Benchmarks")
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
            st.markdown("#### Hedging Profit (Optimal vs 50/50)")
            hp1, hp2, hp3 = st.columns(3)
            hp1.metric("Annual alpha vs 50/50",
                       f"{hp*100:+.2f}%", delta=f"${bal*hp:+,.0f}/yr",
                       delta_color="normal" if hp >= 0 else "inverse")
            hp2.metric("Optimal return", f"{er_opt*100:.2f}%",
                       delta=f"${bal*er_opt:,.0f}/yr", delta_color="normal")
            hp3.metric("50/50 return",  f"{er_5050*100:.2f}%",
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
        pnl_total = pnl_x + pnl_y
        bench_pnl = bal * 0.5 * ret_x + bal * 0.5 * ret_y

        st.markdown("#### P&L Since Last Calibration")
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
        pl4.metric("50/50 bench P&L", f"${bench_pnl:+,.0f}",
                   delta=f"vs optimal: ${pnl_total - bench_pnl:+,.0f}",
                   delta_color="normal" if (pnl_total - bench_pnl) >= 0 else "inverse")

        cal_ts = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(status.get("last_pipeline_time", 0))
        )
        st.caption(
            f"Entry at calibration ({cal_ts}): "
            f"{cfg['ticker_x']} ${x_entry:.2f} · {cfg['ticker_y']} ${y_entry:.2f}  —  "
            f"next price refresh in ~15 min"
        )

    st.divider()


def _options_section(system: AllocationSystem, cfg: dict, status: dict):
    st.markdown("### Options Overlay — Protective Puts")

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

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Total hedge cost", f"${result['total_hedge_cost_dollars']:,.0f}",
               delta=f"{result['total_hedge_cost_pct']*100:.2f}% of portfolio")
    tc2.metric("Equity exposure", f"${result['total_equity_value']:,.0f}")
    tc3.metric("Hedged portfolio value",
               f"${bal - result['total_hedge_cost_dollars']:,.0f}")

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


def _backtest_section(system: AllocationSystem, cfg: dict):
    st.markdown("### Historical Backtest")
    st.caption(
        "Rolls a calibration window forward, rebalances at chosen frequency using N=64 for speed. "
        "All returns are log-return compounded."
    )

    bc1, bc2, bc3 = st.columns(3)
    cal_window = bc1.selectbox("Calibration window (days)", [252, 504, 756], index=1, key="bt_win")
    rebal_freq = bc2.selectbox("Rebalance frequency (days)", [5, 21, 63], index=1, key="bt_rebal")
    bt_gamma = bc3.slider("γ for backtest", -5.0, -0.1, value=cfg["gamma"], step=0.1, key="bt_gamma")

    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None

    if st.button("▶ Run Backtest", key="bt_run"):
        _run_backtest_with_progress(system, cfg, cal_window, rebal_freq, bt_gamma)

    bt: Optional[BacktestResult] = st.session_state.bt_result
    if bt is None:
        st.info("Press **▶ Run Backtest** to run the historical simulation.")
        return

    df = bt.to_dataframe()
    metrics = bt.metrics()

    fig_bt = go.Figure()
    colors = {"portfolio": "#2196F3", "bench_5050": "#FF9800",
              f"all_{cfg['ticker_x']}": "#4CAF50", "cash": "#9E9E9E"}
    labels = {
        "portfolio": f"Optimal (γ={bt_gamma:.1f})",
        "bench_5050": "50/50 fixed",
        f"all_{cfg['ticker_x']}": f"All {cfg['ticker_x']}",
        "cash": "Cash",
    }
    for col, color in colors.items():
        if col in df.columns:
            fig_bt.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", name=labels.get(col, col),
                line=dict(color=color, width=2),
            ))

    for d in bt.recal_dates:
        fig_bt.add_vline(x=d, line_dash="dot", line_color="rgba(100,100,100,0.3)")

    fig_bt.update_layout(
        yaxis_title="Wealth (start = 1.0)",
        height=420,
        margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    if metrics:
        st.markdown("#### Performance Summary")
        mcols = st.columns(len(metrics))
        for col, (key, m) in zip(mcols, metrics.items()):
            label = labels.get(key, key)
            col.markdown(f"**{label}**")
            col.metric("Ann. return", f"{m['ann_return']*100:.1f}%")
            col.metric("Ann. vol", f"{m['ann_vol']*100:.1f}%")
            col.metric("Sharpe", f"{m['sharpe']:.2f}")
            col.metric("Total return", f"{m['total_return']*100:.1f}%")
            col.metric("Max drawdown", f"{m['max_drawdown']*100:.1f}%")

    with st.expander("Allocation over time"):
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=df.index, y=df["pi_x"] * 100, mode="lines",
            name=cfg["ticker_x"], fill="tozeroy", line=dict(color="#2196F3"),
        ))
        fig_alloc.add_trace(go.Scatter(
            x=df.index, y=df["pi_y"] * 100, mode="lines",
            name=cfg["ticker_y"], fill="tozeroy", line=dict(color="#4CAF50"),
        ))
        fig_alloc.update_layout(
            yaxis_title="Allocation (%)", height=250, margin=dict(t=10, b=40),
        )
        st.plotly_chart(fig_alloc, use_container_width=True)


def _run_backtest_with_progress(
    system: AllocationSystem, cfg: dict,
    cal_window: int, rebal_freq: int, bt_gamma: float,
):
    st.markdown("#### Running backtest...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def _cb(done: int, total: int):
        pct = int(done / total * 100)
        progress_bar.progress(pct)
        status_text.caption(f"Rebalance {done}/{total}  ({pct}%)")

    try:
        bt_feed = DataFeed(
            ticker_x=cfg["ticker_x"],
            ticker_y=cfg["ticker_y"],
            lookback_years=5,
        )
        log_rets = bt_feed.log_returns()
        rfr_series = None
        try:
            rfr_val = bt_feed.risk_free_rate()
            rfr_series = pd.Series(rfr_val, index=log_rets.index)
        except Exception:
            pass

        result = run_backtest(
            log_rets,
            risk_free_rates=rfr_series,
            gamma=bt_gamma,
            alpha=cfg["alpha"],
            horizon_years=cfg["horizon"],
            cal_window_days=cal_window,
            rebal_freq_days=rebal_freq,
            ticker_x=cfg["ticker_x"],
            ticker_y=cfg["ticker_y"],
            progress_cb=_cb,
        )
        st.session_state.bt_result = result
        progress_bar.progress(100)
        status_text.success(
            f"Backtest complete — {len(result.recal_dates)} rebalances over {len(log_rets)} days"
        )
    except Exception as e:
        import traceback
        progress_bar.progress(0)
        status_text.error(f"Backtest failed: {e}")
        st.code(traceback.format_exc())


def main():
    _init_state()
    cfg = _sidebar()

    st.title("📈 Live Two-Asset Allocation System")
    st.caption(
        f"Merton jump-diffusion · Bellman solver · CRRA utility · "
        f"{cfg['ticker_x']} / {cfg['ticker_y']}"
    )

    if st.button("▶ Run / Recalibrate", type="primary"):
        _run_pipeline_with_progress(cfg)
        if "pipeline_error" not in st.session_state:
            st.rerun()

    if "pipeline_log" in st.session_state:
        with st.expander("Last pipeline log", expanded=False):
            st.code(st.session_state.pipeline_log)

    if "pipeline_error" in st.session_state:
        st.error("Pipeline error — see log above")
        with st.expander("Full traceback"):
            st.code(st.session_state.pipeline_error)
        if st.button("Clear error"):
            del st.session_state["pipeline_error"]
            st.rerun()

    system: Optional[AllocationSystem] = st.session_state.system
    if system is None:
        st.info("Press **▶ Run / Recalibrate** to start the pipeline.")
        st.stop()

    _live_section(system, cfg)

    st.markdown("### Optimal Policy Surface")
    surface = system.policy_surface()
    if surface is not None:
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
        for ax, z, title, cs, dot_color in [
            (hc1, pi_x_surf,    f"π_x* ({cfg['ticker_x']})", "Blues",   "red"),
            (hc2, pi_y_surf,    f"π_y* ({cfg['ticker_y']})", "Greens",  "red"),
            (hc3, pi_cash_surf, "Cash fraction",              "Oranges", "blue"),
        ]:
            with ax:
                fig = _heatmap(z, surface["x"], surface["y"], title, cs)
                if px_s and py_s:
                    fig.add_scatter(
                        x=[px_s], y=[py_s], mode="markers",
                        marker=dict(color=dot_color, size=10), name="Current",
                    )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    static_status = system.status()

    st.markdown("### Calibrated Model Parameters")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown("**Diffusion**")
        sx = static_status.get("sigma_x")
        sy = static_status.get("sigma_y")
        st.metric(f"σ_x ({cfg['ticker_x']})", f"{sx*100:.1f}%" if sx else "—")
        st.metric(f"σ_y ({cfg['ticker_y']})", f"{sy*100:.1f}%" if sy else "—")
        rho = static_status.get("rho")
        st.metric("ρ (Brownian corr)", f"{rho:.3f}" if rho is not None else "—")
    with mc2:
        st.markdown("**Jumps**")
        lam = static_status.get("lam")
        st.metric("λ (jump intensity / yr)", f"{lam:.2f}" if lam is not None else "—")
        nj = static_status.get("n_jump_days")
        nt = static_status.get("n_total_days")
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
            f"μ_x ({cfg['ticker_x']}) blended",
            f"{mux*100:.1f}%" if mux is not None else "—",
            delta=f"raw {mux_raw*100:.1f}% → prior {mux_pri*100:.1f}%" if mux_raw is not None else None,
            delta_color="off",
        )
        st.metric(
            f"μ_y ({cfg['ticker_y']}) blended",
            f"{muy*100:.1f}%" if muy is not None else "—",
            delta=f"raw {muy_raw*100:.1f}% → prior {muy_pri*100:.1f}%" if muy_raw is not None else None,
            delta_color="off",
        )
        if bx is not None:
            st.caption(f"β_x={bx:.2f}  β_y={by:.2f}  auto shrinkage λ={shrink:.2f}")
        st.caption(f"γ = {cfg['gamma']} · T = {cfg['horizon']} yr · α = {cfg['alpha']}")

    st.markdown("**Policy network**")
    net_ready = static_status.get("net_ready", False)
    if net_ready:
        st.success("KAN ready — 2→4→3 RBF-KAN · residual baseline · MSE-trained")
    else:
        st.info("KAN not yet trained — run pipeline first")

    st.divider()

    st.markdown("### Why is the allocation what it is?")
    ep_x     = static_status.get("equity_premium_x")
    ep_y     = static_status.get("equity_premium_y")
    hurdle_x = static_status.get("hurdle_x")
    hurdle_y = static_status.get("hurdle_y")

    if ep_x is not None:
        r_actual = None
        try:
            r_actual = system._feed.risk_free_rate()
        except Exception:
            pass

        st.markdown(
            "Risky beats cash when equity premium (`μ_real − r`) exceeds the hurdle "
            "`r + 1.5σ²` (approximate condition for γ = −1)."
        )
        dc1, dc2 = st.columns(2)
        for col, label, ep, hurdle in [
            (dc1, cfg["ticker_x"], ep_x, hurdle_x),
            (dc2, cfg["ticker_y"], ep_y, hurdle_y),
        ]:
            if ep is not None and hurdle is not None:
                beats = ep > (hurdle - (r_actual or 0.05))
                col.metric(
                    f"{label} equity premium",
                    f"{ep*100:+.1f}%",
                    delta=f"hurdle {hurdle*100:.1f}%",
                    delta_color="normal" if beats else "inverse",
                )
                col.caption("✅ risky preferred" if beats else "❌ cash preferred — try γ closer to 0")
        if r_actual:
            st.caption(f"Risk-free rate (^IRX): {r_actual*100:.2f}%")

    _options_section(system, cfg, static_status)

    st.divider()

    _backtest_section(system, cfg)

    st.divider()

    dur  = static_status.get("pipeline_duration_s", 0)
    last = static_status.get("last_pipeline_time", 0)
    ts   = time.strftime("%H:%M:%S", time.localtime(last)) if last else "never"
    st.caption(
        f"Last pipeline: {ts} · Duration: {dur:.1f}s · "
        f"KAN policy net: {'ready ✓' if static_status.get('net_ready') else 'not trained'} · "
        f"Prices: ~15 min delayed (yfinance free tier) · auto-refresh every 15 min"
    )


main()
