"""
Portfolio Constructor, Analyzer and Optimizer
Optimized for Streamlit with native caching and batch Yahoo Finance downloads.
"""

import logging
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Suppress warnings and debug logs
logging.getLogger("yfinance").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Batch & Cached)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="📊 Downloading stock prices...")
def fetch_stock_data(tickers: tuple[str, ...], years: int) -> pd.DataFrame:
    """Download historical stock prices concurrently in a single batch."""
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            tickers=list(tickers),
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
            timeout=25,
        )

        if raw.empty:
            return pd.DataFrame()

        # Handle multi-ticker vs single-ticker structures in yfinance
        if "Close" in raw.columns:
            close_data = raw["Close"]
        else:
            close_data = raw

        if isinstance(close_data, pd.Series):
            data = close_data.to_frame(name=tickers[0])
        else:
            data = close_data.copy()

        # Clean null columns and forward-fill intermittent holidays
        data = data.dropna(axis=1, how="all").ffill().dropna()
        return data

    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="📊 Downloading benchmark index...")
def fetch_market_data(years: int) -> pd.Series:
    """Download NIFTY 50 index data for market benchmarking."""
    try:
        raw = yf.download(
            tickers="^NSEI",
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
            timeout=25,
        )

        if raw.empty:
            return pd.Series(dtype=float)

        series = raw["Close"].squeeze() if "Close" in raw.columns else raw.squeeze()
        return series.ffill().dropna()

    except Exception as e:
        st.error(f"❌ Could not download NIFTY 50: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def portfolio_create(
    tickers: tuple[str, ...], weights: tuple[float, ...], data: pd.DataFrame
) -> pd.Series:
    """Calculate portfolio cumulative daily returns."""
    weights_arr = np.array(weights, dtype=float)
    valid_tickers = [t for t in tickers if t in data.columns]

    if not valid_tickers:
        return pd.Series(dtype=float)

    ticker_to_weight = {tickers[i]: weights_arr[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()

    ret = data[valid_tickers].pct_change().dropna()
    if ret.empty:
        return pd.Series(dtype=float)

    portfolio_returns = (ret * valid_weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()
    cum_returns = cum_returns / cum_returns.iloc[0]

    return cum_returns.rename("Portfolio")


@st.cache_data
def calculate_cumulative_returns(
    data: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """Compute base cumulative growth series starting at 1.0."""
    ret = data.pct_change().dropna()
    cum = (1 + ret).cumprod()
    if isinstance(cum, pd.Series):
        return cum / cum.iloc[0]
    return cum.div(cum.iloc[0])


@st.cache_data
def calculate_metrics(
    portfolio_cum: pd.Series,
    market_series: pd.Series,
    years: int,
    risk_free_rate_pct: float = 6.66,
) -> pd.DataFrame:
    """Calculate Sharpe, Beta, Alpha, and annualized returns."""
    if portfolio_cum.empty or market_series.empty:
        return pd.DataFrame()

    portfolio_returns = portfolio_cum.pct_change().dropna()
    market_returns = market_series.pct_change().dropna()

    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns, join="inner"
    )

    if portfolio_returns.empty:
        return pd.DataFrame()

    covariance = portfolio_returns.cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance if market_variance > 0 else 0.0

    market_cagr_pct = (((market_series.iloc[-1] / market_series.iloc[0]) ** (1 / years)) - 1) * 100
    portfolio_cagr_pct = (((portfolio_cum.iloc[-1] / portfolio_cum.iloc[0]) ** (1 / years)) - 1) * 100

    capm_return_pct = risk_free_rate_pct + beta * (market_cagr_pct - risk_free_rate_pct)
    portfolio_volatility_pct = portfolio_returns.std() * np.sqrt(252) * 100

    daily_rf = (1 + risk_free_rate_pct / 100) ** (1 / 252) - 1
    excess_returns = portfolio_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

    correlation = portfolio_returns.corr(market_returns)

    metrics = {
        "Beta": beta,
        "Market CAGR (%)": market_cagr_pct,
        "CAPM Expected Return (%)": capm_return_pct,
        "Actual CAGR (%)": portfolio_cagr_pct,
        "Alpha (%)": portfolio_cagr_pct - capm_return_pct,
        "Annualised Volatility (%)": portfolio_volatility_pct,
        "Sharpe Ratio": sharpe,
        "Correlation with Market": correlation,
    }

    return pd.DataFrame(metrics, index=["Portfolio"]).round(4)


@st.cache_data
def calculate_individual_returns(stock_data: pd.DataFrame, years: int) -> pd.DataFrame:
    """Calculate standalone returns and volatility per constituent."""
    rows = []
    for col in stock_data.columns:
        series = stock_data[col].dropna()
        if len(series) < 2:
            continue

        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        if 1 + total_return <= 0:
            ann_return_pct = -100.0
        else:
            ann_return_pct = (((1 + total_return) ** (1 / years)) - 1) * 100

        volatility_pct = series.pct_change().dropna().std() * np.sqrt(252) * 100
        rows.append(
            {
                "Stock": col,
                "Total Return (%)": round(total_return * 100, 2),
                "Annualised Return (%)": round(ann_return_pct, 2),
                "Volatility (%)": round(volatility_pct, 2),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("Annualised Return (%)", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# OPTIMIZATION (Vectorized Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def run_optimization(cols: tuple[str, ...], values: tuple, n_scenarios: int) -> dict:
    """Run fully vectorized Monte Carlo portfolio simulation."""
    stock_data = pd.DataFrame({c: np.array(v) for c, v in zip(cols, values)})
    actual_tickers = list(stock_data.columns)
    returns = stock_data.pct_change().dropna()

    n_assets = len(actual_tickers)
    mean_daily_ret = returns.mean().values
    cov_daily_matrix = returns.cov().values

    # Generate Dirichlet-distributed or normalized random weights (n_scenarios, n_assets)
    raw_weights = np.random.random((n_scenarios, n_assets))
    weights = raw_weights / raw_weights.sum(axis=1, keepdims=True)

    # Vectorized return, variance, and volatility calculations
    annual_returns = np.dot(weights, mean_daily_ret) * 252
    annual_risks = np.sqrt(np.sum((weights @ (cov_daily_matrix * 252)) * weights, axis=1))

    sharpe_ratios = np.divide(
        annual_returns,
        annual_risks,
        out=np.zeros_like(annual_returns),
        where=annual_risks > 0,
    )

    optimal_idx = int(np.argmax(sharpe_ratios))

    return {
        "tickers": actual_tickers,
        "optimal_idx": optimal_idx,
        "weights": weights,
        "returns": annual_returns,
        "risks": annual_risks,
        "sharpe": sharpe_ratios,
    }


def optimize_portfolio(stock_data: pd.DataFrame, n_scenarios: int) -> None:
    """Render efficient frontier chart and optimal allocations."""
    actual_tickers = list(stock_data.columns)
    cols_tuple = tuple(actual_tickers)
    vals_tuple = tuple(stock_data[col].values for col in actual_tickers)

    with st.spinner(f"🎯 Running {n_scenarios:,} Monte Carlo simulations..."):
        res = run_optimization(cols_tuple, vals_tuple, n_scenarios)

    optimal_idx = res["optimal_idx"]
    opt_weights = res["weights"][optimal_idx]

    # Efficient Frontier Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        res["risks"],
        res["returns"],
        c=res["sharpe"],
        cmap="plasma",
        alpha=0.5,
        s=15,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(
        res["risks"][optimal_idx],
        res["returns"][optimal_idx],
        color="red",
        marker="*",
        s=600,
        label="Optimal (Max Sharpe)",
        zorder=5,
        edgecolors="darkred",
        linewidth=1.5,
    )
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=11)
    ax.set_ylabel("Annualized Expected Return", fontsize=11)
    ax.set_title("Efficient Frontier", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Output optimal metrics and allocation
    st.success("✅ Optimal Portfolio Found")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{res['sharpe'][optimal_idx]:.4f}")
    col2.metric("Expected Return", f"{res['returns'][optimal_idx] * 100:.2f}%")
    col3.metric("Annualized Volatility", f"{res['risks'][optimal_idx] * 100:.2f}%")

    optimal_df = pd.DataFrame(
        {
            "Stock": actual_tickers,
            "Weight (%)": np.round(opt_weights * 100, 2),
        }
    ).sort_values("Weight (%)", ascending=False)
    st.dataframe(optimal_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏦 Portfolio Constructor & Optimizer")
    st.caption("Concurrent data ingestion with vectorized Modern Portfolio Theory (MPT)")

    # Load symbol map
    try:
        sym = pd.read_csv("sym.csv")
        sym.columns = [col.strip().replace(" ", "_") for col in sym.columns]
    except FileNotFoundError:
        st.error("❌ 'sym.csv' not found. Place it in the working directory.")
        return

    required_cols = {"NAME_OF_COMPANY", "SYMBOL", "SECTOR"}
    if not required_cols.issubset(sym.columns):
        st.error(f"❌ sym.csv missing required columns: {required_cols}")
        return

    # Sidebar parameters
    with st.sidebar:
        st.header("⚙️ Settings")
        n_stocks = st.slider("Number of stocks", 2, 20, 5)
        years = st.slider("Data period (years)", 1, 20, 5)
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (%)", value=6.66, min_value=0.0, max_value=20.0, step=0.1
        )
        n_scenarios = st.slider(
            "Optimization scenarios", 500, 10000, 2000, step=500
        )

        st.divider()
        if st.button("🔄 Clear App Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Asset and weight inputs
    st.subheader("📈 Configure Portfolio")
    tickers: list[str] = []
    weights: list[float] = []
    sectors: list[str] = []

    stock_cols = st.columns(n_stocks)
    for i, col in enumerate(stock_cols):
        with col:
            name = st.selectbox(
                f"Stock {i+1}",
                sym["NAME_OF_COMPANY"],
                key=f"stock_{i}",
                label_visibility="collapsed",
            )
            row = sym[sym["NAME_OF_COMPANY"] == name].iloc[0]
            tickers.append(f"{row['SYMBOL']}.NS")
            sectors.append(row["SECTOR"])

    weight_cols = st.columns(n_stocks)
    for i, col in enumerate(weight_cols):
        with col:
            w = st.number_input(
                f"Weight {i+1}",
                min_value=0.0,
                max_value=100.0,
                value=round(100.0 / n_stocks, 1),
                step=1.0,
                key=f"weight_{i}",
                label_visibility="collapsed",
            )
            weights.append(w)

    # Input validations
    if len(set(tickers)) < len(tickers):
        duplicates = list({t for t in tickers if tickers.count(t) > 1})
        st.error(f"❌ Duplicate stocks selected: {duplicates}")
        st.stop()

    total_weight = sum(weights)
    if not np.isclose(total_weight, 100.0, atol=1e-3):
        st.error(f"⚠️ Total weight must equal 100%. Currently: {total_weight:.1f}%")
        st.stop()

    col1, col2, _ = st.columns([1, 1, 2])
    analyze_button = col1.button("📊 Analyse", use_container_width=True)
    optimize_button = col2.button("🎯 Optimize", use_container_width=True)

    if not (analyze_button or optimize_button):
        return

    # Data ingestion
    tickers_tuple = tuple(tickers)
    weights_tuple = tuple(np.array(weights, dtype=float) / 100.0)

    stock_data = fetch_stock_data(tickers_tuple, years)
    market_series = fetch_market_data(years)

    if stock_data.empty or market_series.empty:
        st.error("❌ Failed to fetch historical prices. Verify symbols and connectivity.")
        st.stop()

    valid_tickers = [t for t in tickers if t in stock_data.columns]
    if len(valid_tickers) < 2:
        st.error("❌ Fewer than 2 stocks returned valid historical data.")
        st.stop()

    ticker_to_weight_orig = {tickers[i]: weights_tuple[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight_orig[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()

    ticker_to_sector = {tickers[i]: sectors[i] for i in range(len(tickers))}
    valid_sectors = [ticker_to_sector[t] for t in valid_tickers]

    # Portfolio Analysis Run
    if analyze_button:
        portfolio_cum = portfolio_create(tickers_tuple, weights_tuple, stock_data)
        if portfolio_cum.empty:
            st.error("❌ Portfolio generation failed.")
            st.stop()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Returns", "Metrics", "Constituent Stocks", "Sector Allocation"]
        )

        with tab1:
            market_cum = calculate_cumulative_returns(market_series).rename("NIFTY 50")
            compare_df = pd.concat([portfolio_cum, market_cum], axis=1).dropna()

            fig, ax = plt.subplots(figsize=(12, 5))
            compare_df.plot(ax=ax, linewidth=2)
            ax.set_title("Cumulative Growth (Base ₹1)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Portfolio Multiple")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            metrics_df = calculate_metrics(
                portfolio_cum, market_series, years, risk_free_rate_pct
            )
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)

        with tab3:
            ind_returns = calculate_individual_returns(stock_data[valid_tickers], years)
            if not ind_returns.empty:
                st.dataframe(ind_returns, use_container_width=True)

                fig, ax = plt.subplots(figsize=(12, 5))
                colors = [
                    "#2ecc71" if v >= 0 else "#e74c3c"
                    for v in ind_returns["Annualised Return (%)"]
                ]
                ax.bar(
                    ind_returns["Stock"],
                    ind_returns["Annualised Return (%)"],
                    color=colors,
                )
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title("Annualized Return by Stock", fontsize=13, fontweight="bold")
                ax.set_ylabel("CAGR (%)")
                ax.grid(alpha=0.3, axis="y")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with tab4:
            sector_df = pd.DataFrame({"Sector": valid_sectors, "Weight": valid_weights})
            sector_agg = sector_df.groupby("Sector")["Weight"].sum()

            fig, ax = plt.subplots(figsize=(7, 7))
            colors = plt.cm.Set3(np.linspace(0, 1, len(sector_agg)))
            ax.pie(
                sector_agg,
                labels=sector_agg.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            ax.set_title("Sector Exposure", fontsize=13, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Optimization Run
    if optimize_button:
        optimize_portfolio(stock_data[valid_tickers], n_scenarios)


if __name__ == "__main__":
    main()
