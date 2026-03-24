import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import plotly.graph_objects as go
from live.system import AllocationSystem
from live.backtest import run_backtest, BacktestResult, bootstrap_ci
from live.data_feed import DataFeed
from live.surrogate import (
    SurrogateKAN, generate_training_data, train_surrogate,
    save_surrogate, load_surrogate, surrogate_available,
    validate_surrogate,
    _SURROGATE_PATH, _META_PATH,
)

st.set_page_config(
    page_title="Backtest & Stats",
    page_icon="📊",
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


def _sidebar(system: Optional[AllocationSystem]) -> dict:
    st.sidebar.title("⚙️ Backtest Settings")

    if system is not None:
        st.sidebar.info(
            f"Using calibrated pair: **{system.ticker_x} / {system.ticker_y}**\n\n"
            f"γ = {system.gamma}  ·  α = {system.alpha}  ·  T = {system.horizon_years} yr"
        )
        ticker_x = system.ticker_x
        ticker_y = system.ticker_y
        default_gamma = system.gamma
        default_alpha = system.alpha
        default_horizon = system.horizon_years
    else:
        st.sidebar.warning("Run the model pipeline first (Model page).")
        ticker_x = "SPY"
        ticker_y = "QQQ"
        default_gamma = -1.0
        default_alpha = 0.5
        default_horizon = 1.0

    st.sidebar.markdown("---")
    lookback_years = st.sidebar.selectbox(
        "Data lookback (years)", [5, 10, 15, 20], index=1,
        help=(
            "How many years of history to fetch. "
            "SPY/QQQ: up to 27 yr · TLT/GLD: up to 22 yr · "
            "Longer = tighter CIs but slower backtest."
        ),
    )
    cal_window = st.sidebar.selectbox(
        "Calibration window (days)", [252, 504, 756], index=1,
        help="Rolling window used to calibrate the Merton model at each rebalance date.",
    )
    rebal_freq = st.sidebar.selectbox(
        "Rebalance frequency (days)", [5, 21, 63], index=1,
        help="How often the portfolio is rebalanced to the optimal allocation.",
    )
    bt_gamma = st.sidebar.slider(
        "Risk aversion γ for backtest", -5.0, -0.1,
        value=float(default_gamma), step=0.1,
        help="Can differ from the live model gamma to test sensitivity.",
    )

    approx_rebal = int(lookback_years * 252 / max(rebal_freq, 1))
    st.sidebar.caption(
        f"~{lookback_years * 252:,} trading days · "
        f"~{approx_rebal} rebalances · "
        f"est. {approx_rebal * 30 // 60} – {approx_rebal * 60 // 60} min runtime"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Bootstrap confidence intervals**")
    n_boot = st.sidebar.selectbox(
        "Bootstrap iterations", [200, 500, 1000], index=1,
        help="More iterations = tighter CI estimates. 500 is a good balance.",
    )
    block_len = st.sidebar.selectbox(
        "Block length (days)", [5, 10, 21], index=2,
        help="Stationary block bootstrap block size. 21 ≈ one trading month.",
    )

    return dict(
        ticker_x=ticker_x, ticker_y=ticker_y,
        gamma=bt_gamma, alpha=default_alpha, horizon=default_horizon,
        lookback_years=lookback_years,
        cal_window=cal_window, rebal_freq=rebal_freq,
        n_boot=n_boot, block_len=block_len,
    )


def _build_surrogate_inline(cfg: dict) -> Optional[object]:
    """
    Build, train, and validate the surrogate KAN inline (200 solves).
    Shows a two-phase progress bar.  Returns the loaded SurrogateKAN or raises.
    """
    N_SOLVES = 200
    prog_bar   = st.progress(0)
    phase_text = st.empty()

    phase_text.info(f"Phase 1/3 — Solving Bellman ({N_SOLVES} parameter sets)…")

    def _solve_cb(done, total):
        prog_bar.progress(int(done / total * 55))
        phase_text.info(f"Phase 1/3 — Bellman solve {done}/{total}…")

    inputs, targets = generate_training_data(
        n_solves=N_SOLVES,
        gamma=cfg["gamma"],
        alpha=cfg["alpha"],
        horizon_years=cfg["horizon"],
        progress_cb=_solve_cb,
    )
    phase_text.info(f"Phase 2/3 — Training SurrogateKAN on {len(inputs):,} examples…")
    prog_bar.progress(60)

    net = SurrogateKAN()
    net, first_mse, final_mse = train_surrogate(
        net, inputs, targets, epochs=300, verbose=False,
    )
    prog_bar.progress(85)

    save_surrogate(
        net,
        gamma=cfg["gamma"], alpha=cfg["alpha"],
        horizon_years=cfg["horizon"],
        n_solves=N_SOLVES, train_mse=final_mse,
    )

    prog_bar.progress(100)
    phase_text.success(
        f"Surrogate ready — {N_SOLVES} solves · MSE {first_mse:.4f}→{final_mse:.4f}"
    )
    return net


def _get_surrogate(cfg: dict) -> object:
    """
    Return a ready-to-use SurrogateKAN for the current (γ, α, T).

    Logic:
      - No surrogate on disk → build one.
      - Surrogate exists but params (γ/α/T) don't match → delete and rebuild.
      - Surrogate exists and params match → load and validate.
    """
    if not surrogate_available():
        st.info("No surrogate found — building now (this takes ~4–8 min the first time).")
        return _build_surrogate_inline(cfg)

    # Surrogate exists — check params
    try:
        net, meta = load_surrogate()
    except Exception:
        # Corrupt files — rebuild
        _delete_surrogate_files()
        return _build_surrogate_inline(cfg)

    params_match = (
        abs(meta["gamma"]         - cfg["gamma"])   < 1e-4 and
        abs(meta["alpha"]         - cfg["alpha"])   < 1e-4 and
        abs(meta["horizon_years"] - cfg["horizon"]) < 1e-4
    )
    if not params_match:
        st.info(
            f"Surrogate was trained with γ={meta['gamma']}, α={meta['alpha']}, "
            f"T={meta['horizon_years']} yr — current settings differ. Rebuilding…"
        )
        _delete_surrogate_files()
        return _build_surrogate_inline(cfg)

    # Soft validation — warn if error is large but don't block the backtest
    try:
        _, err = validate_surrogate(net, cfg["gamma"], cfg["alpha"], cfg["horizon"], tol=0.20)
        if err > 0.10:
            st.warning(
                f"Surrogate point-error is {err*100:.1f}pp at the reference point — "
                "results may be approximate. Consider rebuilding with more solves."
            )
    except RuntimeError as e:
        st.warning(str(e))
    return net


def _delete_surrogate_files() -> None:
    import os
    for p in (_SURROGATE_PATH, _META_PATH):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def _run_backtest_with_progress(cfg: dict) -> Optional[BacktestResult]:
    # ── Phase 0: surrogate ────────────────────────────────────────────────
    surrogate = None
    try:
        surrogate = _get_surrogate(cfg)
    except RuntimeError as e:
        st.error(str(e))
        return None
    except Exception as e:
        import traceback
        st.error(f"Surrogate build/load failed: {e}")
        st.code(traceback.format_exc())
        return None

    # ── Phase 1: backtest ─────────────────────────────────────────────────
    st.markdown("#### Running backtest…")
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
            lookback_years=cfg["lookback_years"],
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
            gamma=cfg["gamma"],
            alpha=cfg["alpha"],
            horizon_years=cfg["horizon"],
            cal_window_days=cfg["cal_window"],
            rebal_freq_days=cfg["rebal_freq"],
            ticker_x=cfg["ticker_x"],
            ticker_y=cfg["ticker_y"],
            progress_cb=_cb,
            surrogate=surrogate,
        )
        progress_bar.progress(100)
        status_text.success(
            f"Backtest complete — {len(result.recal_dates)} rebalances over {len(log_rets)} trading days"
        )
        return result
    except Exception as e:
        import traceback
        progress_bar.progress(0)
        status_text.error(f"Backtest failed: {e}")
        st.code(traceback.format_exc())
        return None


def _render_chart(bt: BacktestResult, bt_gamma: float, ticker_x: str):
    df = bt.to_dataframe()
    fig = go.Figure()
    colors = {
        "portfolio":        "#2196F3",
        "bench_5050":       "#FF9800",
        f"all_{ticker_x}":  "#4CAF50",
        "cash":             "#9E9E9E",
    }
    labels = {
        "portfolio":        f"Optimal (γ={bt_gamma:.1f})",
        "bench_5050":       "50/50 fixed",
        f"all_{ticker_x}":  f"All {ticker_x}",
        "cash":             "Cash",
    }
    for col, color in colors.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode="lines",
                name=labels.get(col, col),
                line=dict(color=color, width=2),
            ))
    for d in bt.recal_dates:
        fig.add_vline(x=d, line_dash="dot", line_color="rgba(150,150,150,0.2)")
    fig.update_layout(
        yaxis_title="Wealth (start = 1.0)",
        height=420,
        margin=dict(t=10, b=40),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


def _render_metrics_with_ci(bt: BacktestResult, cfg: dict, ci: dict):
    import pandas as pd
    metrics = bt.metrics()
    if not metrics:
        return

    ticker_x = cfg["ticker_x"]
    labels = {
        "portfolio": f"Optimal (γ={cfg['gamma']:.1f})",
        "bench_5050": "50/50 fixed",
        f"all_{ticker_x}": f"All {ticker_x}",
        "cash": "Cash",
    }

    # ── Performance table ───────────────────────────────────────────────────
    st.markdown("### Performance Summary")
    rows = []
    for key, m in metrics.items():
        rows.append({
            "Strategy":     labels.get(key, key),
            "Ann. Return":  f"{m['ann_return']*100:.1f}%",
            "Ann. Vol":     f"{m['ann_vol']*100:.1f}%",
            "Sharpe":       f"{m['sharpe']:.2f}",
            "Total Return": f"{m['total_return']*100:.1f}%",
            "Max DD":       f"{m['max_drawdown']*100:.1f}%",
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Strategy"),
        use_container_width=True,
    )

    if not ci:
        return

    # ── Bootstrap CI ────────────────────────────────────────────────────────
    with st.expander("Bootstrap Confidence Intervals (95%, stationary block bootstrap)", expanded=True):
        st.caption(
            "Stationary block bootstrap resamples daily returns in 21-day blocks "
            "to preserve autocorrelation. 500 resampled paths."
        )
        ci_rows = []
        for key, m in metrics.items():
            label = labels.get(key, key)
            c = ci.get(key, {})
            row = {"Strategy": label}
            for metric_key, metric_label, fmt in [
                ("ann_return",    "Ann. Return", lambda v: f"{v*100:.1f}%"),
                ("sharpe",        "Sharpe",      lambda v: f"{v:.2f}"),
                ("max_drawdown",  "Max DD",      lambda v: f"{v*100:.1f}%"),
            ]:
                if metric_key in c:
                    lo = c[metric_key]["ci_low"]
                    hi = c[metric_key]["ci_high"]
                    mn = c[metric_key]["mean"]
                    row[metric_label] = f"{fmt(mn)}  [{fmt(lo)}, {fmt(hi)}]"
                else:
                    row[metric_label] = "—"
            ci_rows.append(row)

        ci_df = pd.DataFrame(ci_rows).set_index("Strategy")
        st.dataframe(ci_df, use_container_width=True)

        with st.expander("Statistical test: Is optimal Sharpe significantly > 50/50?"):
            port_sharpes = []
            bench_sharpes = []

            port_lr  = np.log(np.maximum(bt.portfolio[1:] / bt.portfolio[:-1], 1e-30))
            bench_lr = np.log(np.maximum(bt.bench_5050[1:] / bt.bench_5050[:-1], 1e-30))
            n = len(port_lr)
            block_len = cfg.get("block_len", 21)
            rng = np.random.default_rng(99)
            n_blocks = int(np.ceil(n / block_len))
            max_start = max(n - block_len, 1)

            for _ in range(cfg.get("n_boot", 500)):
                starts = rng.integers(0, max_start + 1, size=n_blocks)
                idx = np.concatenate([
                    np.arange(s, min(s + block_len, n)) for s in starts
                ])[:n]
                blr = port_lr[idx]
                bmu = blr.mean() * 252; bvol = blr.std() * np.sqrt(252)
                port_sharpes.append(bmu / bvol if bvol > 1e-9 else 0.0)
                blr2 = bench_lr[idx]
                bmu2 = blr2.mean() * 252; bvol2 = blr2.std() * np.sqrt(252)
                bench_sharpes.append(bmu2 / bvol2 if bvol2 > 1e-9 else 0.0)

            diff    = np.array(port_sharpes) - np.array(bench_sharpes)
            p_val   = float(np.mean(diff <= 0))
            opt_sharpe   = metrics.get("portfolio", {}).get("sharpe", 0.0)
            bench_sharpe = metrics.get("bench_5050", {}).get("sharpe", 0.0)
            diff_pt      = opt_sharpe - bench_sharpe

            r1, r2, r3 = st.columns(3)
            r1.metric("Optimal Sharpe", f"{opt_sharpe:.2f}")
            r2.metric("50/50 Sharpe",   f"{bench_sharpe:.2f}")
            r3.metric("Difference",     f"{diff_pt:+.2f}")

            ci_lo = float(np.quantile(diff, 0.025))
            ci_hi = float(np.quantile(diff, 0.975))
            st.markdown(
                f"**Sharpe difference (Optimal − 50/50):** "
                f"{diff_pt:+.2f}  &nbsp; 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]",
                unsafe_allow_html=True,
            )
            if p_val < 0.05:
                st.success(f"p = {p_val:.3f} < 0.05 — optimal Sharpe is **statistically significantly higher** than 50/50.")
            elif p_val < 0.10:
                st.warning(f"p = {p_val:.3f} — marginal evidence (10% level) of outperformance.")
            else:
                st.info(f"p = {p_val:.3f} — no statistically significant Sharpe outperformance vs 50/50.")
            st.caption("H₀: Sharpe(optimal) ≤ Sharpe(50/50). p = fraction of bootstrap samples where difference ≤ 0.")


def main():
    st.markdown("## 📊 Historical Backtest & Statistical Tests")

    system: Optional[AllocationSystem] = st.session_state.get("system")
    cfg = _sidebar(system)

    # ── Surrogate status banner ─────────────────────────────────────────────
    if surrogate_available():
        try:
            _, meta = load_surrogate()
            params_match = (
                abs(meta["gamma"]         - cfg["gamma"])   < 1e-4 and
                abs(meta["alpha"]         - cfg["alpha"])   < 1e-4 and
                abs(meta["horizon_years"] - cfg["horizon"]) < 1e-4
            )
            with st.container(border=True):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Surrogate solves", meta["n_solves"])
                s2.metric("Train MSE",        f"{meta['train_mse']:.5f}")
                s3.metric("γ / α / T",        f"{meta['gamma']} / {meta['alpha']} / {meta['horizon_years']}")
                s4.metric("Params match",     "✅ Yes" if params_match else "⚠️ Mismatch — will rebuild")
        except Exception:
            pass
    else:
        st.info("No surrogate on disk — will build automatically on first run (~8 min).")

    # ── Run buttons ─────────────────────────────────────────────────────────
    if "bt_result" not in st.session_state:
        st.session_state.bt_result = None
    if "bt_ci" not in st.session_state:
        st.session_state.bt_ci = None

    col_run, col_ci = st.columns([1, 1])
    run_bt = col_run.button("▶ Run Backtest", type="primary", use_container_width=True)
    run_ci = col_ci.button(
        "🔁 Run Bootstrap CI",
        help="Run after backtest completes (~30s for 500 iterations).",
        disabled=(st.session_state.bt_result is None),
        use_container_width=True,
    )

    if run_bt:
        result = _run_backtest_with_progress(cfg)
        if result is not None:
            st.session_state.bt_result = result
            st.session_state.bt_ci = None
            st.rerun()

    bt: Optional[BacktestResult] = st.session_state.bt_result
    if bt is None:
        st.info("Configure the sidebar and press **▶ Run Backtest** to start.")
        st.stop()

    if run_ci:
        with st.spinner(f"Running {cfg['n_boot']} bootstrap iterations..."):
            ci = bootstrap_ci(bt, n_boot=cfg["n_boot"], block_len=cfg["block_len"])
            st.session_state.bt_ci = ci
        st.rerun()

    # ── Wealth chart ────────────────────────────────────────────────────────
    st.plotly_chart(
        _render_chart(bt, cfg["gamma"], cfg["ticker_x"]),
        use_container_width=True,
    )

    # ── Performance & CI ────────────────────────────────────────────────────
    ci_data = st.session_state.bt_ci
    _render_metrics_with_ci(bt, cfg, ci_data or {})

    if ci_data is None:
        st.caption("Press **🔁 Run Bootstrap CI** to compute confidence intervals and the Sharpe significance test.")

    # ── Allocation over time ────────────────────────────────────────────────
    with st.expander("Allocation over time"):
        df = bt.to_dataframe()
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=df.index, y=df["pi_x"] * 100, mode="lines",
            name=cfg["ticker_x"], fill="tozeroy",
            line=dict(color="#2196F3", width=1.5),
        ))
        fig_alloc.add_trace(go.Scatter(
            x=df.index, y=df["pi_y"] * 100, mode="lines",
            name=cfg["ticker_y"], fill="tozeroy",
            line=dict(color="#4CAF50", width=1.5),
        ))
        fig_alloc.update_layout(
            yaxis_title="Allocation (%)",
            height=260,
            margin=dict(t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    st.caption(
        f"Lookback: {cfg['lookback_years']} yr · "
        f"Cal. window: {cfg['cal_window']} days · "
        f"Rebal: {cfg['rebal_freq']} days · "
        f"Surrogate KAN: 9→24→12→3, 200 Bellman solves · "
        f"Bootstrap: block={cfg['block_len']} days, n={cfg['n_boot']}"
    )


main()
