"""
Portfolio Constructor, Analyzer and Optimizer
Clean, reliable Streamlit implementation with unified batch data fetching.
"""

import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Suppress noisy logging
logging.getLogger("yfinance").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persist action state across tab changes and reruns
if "action" not in st.session_state:
    st.session_state.action = None


# ─────────────────────────────────────────────────────────────────────────
# UNIFIED DATA INGESTION (Single request, no thread leaks, natively cached)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="📊 Downloading market data from Yahoo Finance...")
def fetch_all_market_data(stock_tickers: tuple[str, ...], benchmark: str, years: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Downloads stock tickers AND the benchmark in ONE single network call.
    threads=False prevents worker thread deadlocks inside Streamlit.
    """
    all_symbols = list(stock_tickers) + [benchmark]
    
    try:
        raw = yf.download(
            tickers=all_symbols,
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
            threads=False,  # Critical: prevents thread deadlocks in Streamlit
            timeout=20,
        )

        if raw.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Extract 'Close' prices from single-index or multi-index frames
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                close_df = raw["Close"].copy()
            else:
                close_df = raw.xs(raw.columns.levels[0][0], axis=1).copy()
        else:
            if "Close" in raw.columns:
                close_df = raw[["Close"]].copy()
            else:
                close_df = raw.copy()

        # Clean gaps
        close_df = close_df.dropna(axis=1, how="all").ffill().bfill()

        # Separate benchmark from individual stocks
        if benchmark in close_df.columns:
            benchmark_series = close_df[benchmark].dropna().rename("NIFTY 50")
            stock_df = close_df.drop(columns=[benchmark], errors="ignore")
        else:
            benchmark_series = pd.Series(dtype=float)
            stock_df = close_df

        return stock_df, benchmark_series

    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return pd.DataFrame(), pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────
# PORTFOLIO CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────

def calculate_portfolio_cumulative(data: pd.DataFrame, tickers: list[str], weights: np.ndarray) -> pd.Series:
    """Calculate cumulative normalized portfolio returns (starts at 1.0)."""
    valid = [t for t in tickers if t in data.columns]
    if not valid:
        return pd.Series(dtype=float)

    weight_map = dict(zip(tickers, weights))
    w = np.array([weight_map[t] for t in valid], dtype=float)
    w /= w.sum()

    daily_ret = data[valid].pct_change().dropna()
    if daily_ret.empty:
        return pd.Series(dtype=float)

    port_daily = (daily_ret * w).sum(axis=1)
    cum = (1 + port_daily).cumprod()
    return (cum / cum.iloc[0]).rename("Portfolio")


def calculate_metrics(portfolio_cum: pd.Series, market_series: pd.Series, years: int, rf_pct: float) -> pd.DataFrame:
    """Calculate Beta, Alpha, Sharpe, and CAGR."""
    port_ret = portfolio_cum.pct_change().dropna()
    mkt_ret = market_series.pct_change().dropna()
    port_ret, mkt_ret = port_ret.align(mkt_ret, join="inner")

    if len(port_ret) < 2:
        return pd.DataFrame()

    cov = port_ret.cov(mkt_ret)
    mkt_var = mkt_ret.var()
    beta = cov / mkt_var if mkt_var > 0 else 0.0

    mkt_cagr = (((market_series.iloc[-1] / market_series.iloc[0]) ** (1 / years)) - 1) * 100
    port_cagr = (((portfolio_cum.iloc[-1] / portfolio_cum.iloc[0]) ** (1 / years)) - 1) * 100
    capm_expected = rf_pct + beta * (mkt_cagr - rf_pct)
    volatility = port_ret.std() * np.sqrt(252) * 100

    daily_rf = (1 + rf_pct / 100) ** (1 / 252) - 1
    excess_ret = port_ret - daily_rf
    sharpe = (excess_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() > 0 else 0.0

    return pd.DataFrame(
        {
            "Beta": beta,
            "Market CAGR (%)": mkt_cagr,
            "CAPM Expected Return (%)": capm_expected,
            "Actual CAGR (%)": port_cagr,
            "Alpha (%)": port_cagr - capm_expected,
            "Annualised Volatility (%)": volatility,
            "Sharpe Ratio": sharpe,
            "Correlation with Market": port_ret.corr(mkt_ret),
        },
        index=["Portfolio"],
    ).round(4)


def calculate_individual_stocks(data: pd.DataFrame, years: int) -> pd.DataFrame:
    """Calculate return and volatility per constituent."""
    rows = []
    for col in data.columns:
        s = data[col].dropna()
        if len(s) < 2:
            continue
        tot_ret = (s.iloc[-1] / s.iloc[0]) - 1
        cagr = (((1 + max(tot_ret, -0.999)) ** (1 / years)) - 1) * 100
        vol = s.pct_change().dropna().std() * np.sqrt(252) * 100
        rows.append({
            "Stock": col,
            "Total Return (%)": round(tot_ret * 100, 2),
            "Annualised Return (%)": round(cagr, 2),
            "Volatility (%)": round(vol, 2),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Annualised Return (%)", ascending=False).reset_index(drop=True) if not df.empty else df


# ─────────────────────────────────────────────────────────────────────────
# MONTE CARLO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────

def run_monte_carlo(stock_data: pd.DataFrame, n_scenarios: int) -> dict:
    """Standard Monte Carlo simulation to plot the efficient frontier."""
    tickers = list(stock_data.columns)
    returns = stock_data.pct_change().dropna()
    mean_ret = returns.mean().values * 252
    cov = returns.cov().values * 252
    n_assets = len(tickers)

    w_all = np.random.random((n_scenarios, n_assets))
    w_all /= w_all.sum(axis=1, keepdims=True)

    p_returns = np.dot(w_all, mean_ret)
    p_risks = np.sqrt(np.sum((w_all @ cov) * w_all, axis=1))
    sharpes = np.where(p_risks > 0, p_returns / p_risks, 0.0)

    best_idx = int(np.argmax(sharpes))
    return {
        "tickers": tickers,
        "best_idx": best_idx,
        "weights": w_all,
        "returns": p_returns,
        "risks": p_risks,
        "sharpes": sharpes,
    }


def display_optimization(stock_data: pd.DataFrame, n_scenarios: int) -> None:
    """Render Frontier chart and weights."""
    with st.spinner(f"Running {n_scenarios:,} simulations..."):
        res = run_monte_carlo(stock_data, n_scenarios)

    idx = res["best_idx"]
    opt_weights = res["weights"][idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(res["risks"], res["returns"] * 100, c=res["sharpes"], cmap="plasma", alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(res["risks"][idx], res["returns"][idx] * 100, color="red", marker="*", s=400, label="Max Sharpe")
    ax.set_xlabel("Annualized Volatility (Risk)")
    ax.set_ylabel("Expected Return (CAGR %)")
    ax.set_title("Efficient Frontier", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.success("✅ Optimal Allocation (Maximum Sharpe)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sharpe Ratio", f"{res['sharpes'][idx]:.4f}")
    c2.metric("Expected Return", f"{res['returns'][idx] * 100:.2f}%")
    c3.metric("Volatility", f"{res['risks'][idx] * 100:.2f}%")

    opt_df = pd.DataFrame({
        "Stock": res["tickers"],
        "Weight (%)": np.round(opt_weights * 100, 2),
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)
    st.dataframe(opt_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏦 Portfolio Constructor & Optimizer")

    # Load symbol metadata
    try:
        sym = pd.read_csv("sym.csv")
        sym.columns = [c.strip().replace(" ", "_") for c in sym.columns]
    except FileNotFoundError:
        st.error("❌ 'sym.csv' file not found in directory.")
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("⚙️ Parameters")
        n_stocks = st.slider("Number of Stocks", min_value=2, max_value=12, value=5)
        years = st.slider("Lookback Period (Years)", min_value=1, max_value=15, value=5)
        risk_free_rate = st.number_input("Risk-free rate (%)", value=6.66, step=0.1)
        n_scenarios = st.slider("Monte Carlo Scenarios", 500, 10000, 2000, step=500)

        st.divider()
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.session_state.action = None
            st.rerun()

    # Asset pickers
    st.subheader("Configure Holdings")
    tickers, weights, sectors = [], [], []
    default_w = round(100.0 / n_stocks, 1)

    cols = st.columns(n_stocks)
    for i, col in enumerate(cols):
        with col:
            name = st.selectbox(
                f"Stock {i+1}",
                sym["NAME_OF_COMPANY"],
                index=i % len(sym),
                key=f"stock_{i}",
            )
            row = sym[sym["NAME_OF_COMPANY"] == name].iloc[0]
            # Strip whitespace to avoid creating invalid Yahoo tickers like 'TCS  .NS'
            symbol_clean = str(row["SYMBOL"]).strip()
            tickers.append(f"{symbol_clean}.NS")
            sectors.append(row["SECTOR"])

    w_cols = st.columns(n_stocks)
    for i, col in enumerate(w_cols):
        with col:
            w = st.number_input(
                f"Weight {i+1} (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_w,
                step=1.0,
                key=f"weight_{i}",
            )
            weights.append(w)

    # Validations
    if len(set(tickers)) < len(tickers):
        st.error("❌ Duplicate stocks selected. Please choose distinct companies.")
        return

    total_w = sum(weights)
    if not np.isclose(total_w, 100.0, atol=0.5):
        st.error(f"⚠️ Total weight is {total_w:.1f}%. It must sum to 100%.")
        return

    # Action buttons that set persistent session state
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    if btn_col1.button("📊 Analyse", use_container_width=True, type="primary"):
        st.session_state.action = "analyse"
    if btn_col2.button("🎯 Optimize", use_container_width=True):
        st.session_state.action = "optimize"

    if st.session_state.action is None:
        return

    # Single unified download for all stocks + benchmark
    stock_data, benchmark_series = fetch_all_market_data(
        stock_tickers=tuple(tickers),
        benchmark="^NSEI",
        years=years,
    )

    if stock_data.empty or benchmark_series.empty:
        st.error("❌ Failed to retrieve price data. Check ticker names or connection.")
        return

    valid_tickers = [t for t in tickers if t in stock_data.columns]
    if len(valid_tickers) < 2:
        st.error("❌ At least 2 active stocks are required.")
        return

    # Normalise weights for active valid tickers
    w_map = dict(zip(tickers, [w / 100.0 for w in weights]))
    active_weights = np.array([w_map[t] for t in valid_tickers], dtype=float)
    active_weights /= active_weights.sum()

    sec_map = dict(zip(tickers, sectors))
    active_sectors = [sec_map[t] for t in valid_tickers]

    # Render Analysis
    if st.session_state.action == "analyse":
        port_cum = calculate_portfolio_cumulative(stock_data, valid_tickers, active_weights)
        if port_cum.empty:
            st.error("❌ Unable to calculate portfolio returns.")
            return

        tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Metrics", "Constituents", "Sectors"])

        with tab1:
            mkt_cum = (1 + benchmark_series.pct_change().dropna()).cumprod()
            mkt_cum = mkt_cum / mkt_cum.iloc[0]
            cmp_df = pd.concat([port_cum, mkt_cum], axis=1).dropna()

            fig, ax = plt.subplots(figsize=(11, 4.5))
            cmp_df.plot(ax=ax, linewidth=2)
            ax.set_title("Cumulative Growth (Base ₹1)", fontweight="bold")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            metrics_df = calculate_metrics(port_cum, benchmark_series, years, risk_free_rate)
            st.dataframe(metrics_df, use_container_width=True)

        with tab3:
            ind_df = calculate_individual_stocks(stock_data[valid_tickers], years)
            st.dataframe(ind_df, use_container_width=True)

        with tab4:
            s_df = pd.DataFrame({"Sector": active_sectors, "Weight": active_weights})
            s_agg = s_df.groupby("Sector")["Weight"].sum()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(s_agg, labels=s_agg.index, autopct="%1.1f%%", startangle=90)
            ax.set_title("Sector Exposure", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Render Optimization
    elif st.session_state.action == "optimize":
        display_optimization(stock_data[valid_tickers], n_scenarios)


if __name__ == "__main__":
    main()
