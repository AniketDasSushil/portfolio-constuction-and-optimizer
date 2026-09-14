import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import logging
import warnings

# Suppress debug output and warnings
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


# ─────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Batch downloads + Native Streamlit Caching)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="📊 Fetching stock data from Yahoo Finance...")
def fetch_stock_data(tickers: tuple, years: int) -> pd.DataFrame:
    """
    Download price histories in a single concurrent batch call.
    Uses native Streamlit memory caching to prevent SQLite deadlocks.
    """
    ticker_list = list(tickers)
    try:
        raw = yf.download(
            tickers=ticker_list,
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
            timeout=25,
        )

        if raw.empty:
            return pd.DataFrame()

        # Handle modern yfinance MultiIndex output vs SingleIndex output
        if "Close" in raw.columns:
            close_data = raw["Close"]
            if isinstance(close_data, pd.Series):
                data = close_data.to_frame(name=ticker_list[0])
            else:
                data = close_data
        else:
            data = raw

        # Clean forward/backward fills and drop inactive tickers
        data = data.dropna(axis=1, how="all").ffill().bfill().dropna()
        return data

    except Exception as e:
        st.error(f"❌ Error fetching stock data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner="📊 Fetching benchmark data (^NSEI)...")
def fetch_market_data(years: int) -> pd.Series:
    """Download NIFTY 50 benchmark data with automatic caching."""
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

        if "Close" in raw.columns:
            series = raw["Close"].squeeze()
        else:
            series = raw.squeeze()

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        return series.ffill().bfill().dropna().rename("NIFTY 50")

    except Exception as e:
        st.error(f"❌ Could not download NIFTY 50: {str(e)}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────
# PORTFOLIO CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def portfolio_create(
    tickers: tuple, weights: tuple, data: pd.DataFrame
) -> pd.Series:
    """Calculate normalized portfolio cumulative returns."""
    valid_tickers = [t for t in tickers if t in data.columns]
    if not valid_tickers:
        return pd.Series(dtype=float)

    ticker_to_weight = dict(zip(tickers, weights))
    valid_weights = np.array([ticker_to_weight[t] for t in valid_tickers], dtype=float)
    valid_weights /= valid_weights.sum()

    ret = data[valid_tickers].pct_change().dropna()
    if ret.empty:
        return pd.Series(dtype=float)

    portfolio_daily_returns = (ret * valid_weights).sum(axis=1)
    cum_returns = (1 + portfolio_daily_returns).cumprod()
    return (cum_returns / cum_returns.iloc[0]).rename("Portfolio")


@st.cache_data
def calculate_cumulative_returns(series: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a price series."""
    ret = series.pct_change().dropna()
    cum = (1 + ret).cumprod()
    return cum / cum.iloc[0]


@st.cache_data
def calculate_metrics(
    portfolio_cum: pd.Series,
    market_series: pd.Series,
    years: int,
    risk_free_rate_pct: float = 6.66,
) -> pd.DataFrame:
    """Calculate portfolio financial statistics, CAPM Alpha, Beta, and Sharpe."""
    if portfolio_cum.empty or market_series.empty:
        return pd.DataFrame()

    port_daily_ret = portfolio_cum.pct_change().dropna()
    mkt_daily_ret = market_series.pct_change().dropna()

    port_daily_ret, mkt_daily_ret = port_daily_ret.align(mkt_daily_ret, join="inner")
    if len(port_daily_ret) < 2:
        return pd.DataFrame()

    covariance = port_daily_ret.cov(mkt_daily_ret)
    market_variance = mkt_daily_ret.var()
    beta = covariance / market_variance if market_variance > 0 else 0.0

    market_cagr_pct = (((market_series.iloc[-1] / market_series.iloc[0]) ** (1 / years)) - 1) * 100
    portfolio_cagr_pct = (((portfolio_cum.iloc[-1] / portfolio_cum.iloc[0]) ** (1 / years)) - 1) * 100

    capm_return_pct = risk_free_rate_pct + beta * (market_cagr_pct - risk_free_rate_pct)
    portfolio_volatility_pct = port_daily_ret.std() * np.sqrt(252) * 100

    daily_rf = (1 + risk_free_rate_pct / 100) ** (1 / 252) - 1
    excess_returns = port_daily_ret - daily_rf
    sharpe = (excess_returns.mean() / port_daily_ret.std()) * np.sqrt(252) if port_daily_ret.std() > 0 else 0.0

    correlation = port_daily_ret.corr(mkt_daily_ret)

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
    """Calculate total return, CAGR, and volatility per stock."""
    rows = []
    for col in stock_data.columns:
        series = stock_data[col].dropna()
        if len(series) < 2:
            continue

        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        ann_return_pct = (((1 + max(total_return, -0.9999)) ** (1 / years)) - 1) * 100
        volatility_pct = series.pct_change().dropna().std() * np.sqrt(252) * 100

        rows.append({
            "Stock": col,
            "Total Return (%)": round(total_return * 100, 2),
            "Annualised Return (%)": round(ann_return_pct, 2),
            "Volatility (%)": round(volatility_pct, 2),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values("Annualised Return (%)", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# VECTORIZED MONTE CARLO OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def run_optimization(stock_data: pd.DataFrame, n_scenarios: int) -> dict:
    """
    Vectorized Monte Carlo optimization using matrix operations.
    Executes in milliseconds compared to iterative Python loops.
    """
    tickers = list(stock_data.columns)
    n_assets = len(tickers)
    returns = stock_data.pct_change().dropna()

    mean_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252

    # Generate random weights in bulk
    weights_matrix = np.random.random((n_scenarios, n_assets))
    weights_matrix /= weights_matrix.sum(axis=1, keepdims=True)

    # Vectorized return, volatility, and Sharpe computations
    portfolio_returns = np.sum(weights_matrix * mean_returns, axis=1)
    portfolio_risks = np.sqrt(np.einsum("ij,jk,ik->i", weights_matrix, cov_matrix, weights_matrix))
    sharpe_ratios = np.where(portfolio_risks > 0, portfolio_returns / portfolio_risks, 0.0)

    optimal_idx = int(np.argmax(sharpe_ratios))

    return {
        "tickers": tickers,
        "optimal_idx": optimal_idx,
        "weights": weights_matrix,
        "returns": portfolio_returns,
        "risks": portfolio_risks,
        "sharpe": sharpe_ratios,
    }


def optimize_portfolio(stock_data: pd.DataFrame, n_scenarios: int) -> None:
    """Run optimization and display frontier graph and allocation breakdown."""
    st.write(f"🎯 Running **{n_scenarios:,}** vectorized simulations...")
    opt_result = run_optimization(stock_data, n_scenarios)

    tickers = opt_result["tickers"]
    optimal_idx = opt_result["optimal_idx"]
    risks = opt_result["risks"]
    returns = opt_result["returns"]
    sharpe = opt_result["sharpe"]
    optimal_weights = opt_result["weights"][optimal_idx]

    # Efficient Frontier Scatter Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(
        risks,
        returns * 100,
        c=sharpe,
        cmap="viridis",
        alpha=0.6,
        s=18,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    ax.scatter(
        risks[optimal_idx],
        returns[optimal_idx] * 100,
        color="red",
        marker="*",
        s=500,
        label="Optimal (Max Sharpe)",
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
    )
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=11)
    ax.set_ylabel("Expected Return (CAGR %)", fontsize=11)
    ax.set_title("Markowitz Efficient Frontier", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Optimal Allocation Table
    st.success("✅ Optimal Portfolio Allocation (Maximum Sharpe Ratio)")
    optimal_df = pd.DataFrame({
        "Stock": tickers,
        "Optimal Weight (%)": np.round(optimal_weights * 100, 2),
    }).sort_values("Optimal Weight (%)", ascending=False).reset_index(drop=True)

    col_tbl, col_metrics = st.columns([1, 1])
    with col_tbl:
        st.dataframe(optimal_df, use_container_width=True)

    with col_metrics:
        st.metric("Optimal Sharpe Ratio", f"{sharpe[optimal_idx]:.4f}")
        st.metric("Expected Annual Return", f"{returns[optimal_idx] * 100:.2f}%")
        st.metric("Expected Volatility", f"{risks[optimal_idx] * 100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏦 Portfolio Constructor & Optimizer")
    st.caption("Perform rapid portfolio risk-adjusted return analysis and frontier optimization.")

    # Load symbol universe
    try:
        sym = pd.read_csv("sym.csv")
        sym.columns = [col.strip().replace(" ", "_") for col in sym.columns]
    except FileNotFoundError:
        st.error("❌ `sym.csv` was not found. Please place `sym.csv` in the root working directory.")
        return

    required_cols = {"NAME_OF_COMPANY", "SYMBOL", "SECTOR"}
    if not required_cols.issubset(sym.columns):
        st.error(f"❌ `sym.csv` is missing required columns: {required_cols - set(sym.columns)}")
        return

    # Sidebar parameters
    with st.sidebar:
        st.header("⚙️ Settings")
        n_stocks = st.slider("Number of stocks", min_value=2, max_value=15, value=5)
        years = st.slider("Historical Data (Years)", min_value=1, max_value=15, value=5)
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (%)", value=6.66, min_value=0.0, max_value=20.0, step=0.1
        )
        n_scenarios = st.select_slider(
            "Monte Carlo Scenarios", options=[1000, 2500, 5000, 10000, 20000], value=5000
        )

        st.divider()
        if st.button("🔄 Clear App Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Asset selector and weight input
    st.subheader("📈 Build Portfolio")
    tickers, weights, sectors = [], [], []

    company_names = sym["NAME_OF_COMPANY"].tolist()
    default_weight = round(100.0 / n_stocks, 1)

    stock_cols = st.columns(n_stocks)
    for i, col in enumerate(stock_cols):
        with col:
            name = st.selectbox(
                f"Stock {i+1}",
                options=company_names,
                index=i % len(company_names),
                key=f"stock_select_{i}",
            )
            row = sym[sym["NAME_OF_COMPANY"] == name].iloc[0]
            tickers.append(f"{row['SYMBOL']}.NS")
            sectors.append(row["SECTOR"])

    weight_cols = st.columns(n_stocks)
    for i, col in enumerate(weight_cols):
        with col:
            w = st.number_input(
                f"Weight %",
                min_value=0.0,
                max_value=100.0,
                value=default_weight,
                step=1.0,
                key=f"weight_input_{i}",
            )
            weights.append(w)

    # Pre-execution validation
    if len(set(tickers)) < len(tickers):
        st.error("❌ Duplicate stocks detected. Please select distinct assets.")
        return

    total_weight = sum(weights)
    if not np.isclose(total_weight, 100.0, atol=0.5):
        st.warning(f"⚠️ Total allocation equals **{total_weight:.1f}%**. Target is **100.0%**.")
        return

    # Trigger action buttons
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    analyze_clicked = btn_col1.button("📊 Analyse Portfolio", use_container_width=True, type="primary")
    optimize_clicked = btn_col2.button("🎯 Optimize Weights", use_container_width=True)

    if not (analyze_clicked or optimize_clicked):
        return

    # Execute data download
    tickers_tuple = tuple(tickers)
    normalized_weights = tuple(np.array(weights, dtype=float) / 100.0)

    stock_data = fetch_stock_data(tickers_tuple, years)
    market_series = fetch_market_data(years)

    if stock_data.empty or market_series.empty:
        st.error("❌ Failed to fetch historical prices. Verify symbol tickers and internet connectivity.")
        return

    valid_tickers = [t for t in tickers if t in stock_data.columns]
    if len(valid_tickers) < 2:
        st.error("❌ Need at least 2 valid stocks with active price series.")
        return

    ticker_to_weight = dict(zip(tickers, normalized_weights))
    valid_weights = np.array([ticker_to_weight[t] for t in valid_tickers], dtype=float)
    valid_weights /= valid_weights.sum()

    ticker_to_sector = dict(zip(tickers, sectors))
    valid_sectors = [ticker_to_sector[t] for t in valid_tickers]

    # Portfolio Analysis Workflow
    if analyze_clicked:
        portfolio_cum = portfolio_create(tickers_tuple, normalized_weights, stock_data)
        if portfolio_cum.empty:
            st.error("❌ Unable to calculate portfolio performance.")
            return

        tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Metrics", "Stocks", "Allocations"])

        with tab1:
            market_cum = calculate_cumulative_returns(market_series)
            compare_df = pd.concat([portfolio_cum, market_cum], axis=1).dropna()

            fig, ax = plt.subplots(figsize=(11, 4.5))
            compare_df.plot(ax=ax, linewidth=2.0)
            ax.set_title("Cumulative Growth (Base ₹1.00)", fontsize=13, fontweight="bold")
            ax.set_ylabel("Growth Factor")
            ax.axhline(1, color="grey", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab2:
            metrics_df = calculate_metrics(portfolio_cum, market_series, years, risk_free_rate_pct)
            if not metrics_df.empty:
                st.dataframe(metrics_df, use_container_width=True)

        with tab3:
            ind_returns = calculate_individual_returns(stock_data[valid_tickers], years)
            if not ind_returns.empty:
                st.dataframe(ind_returns, use_container_width=True)

                fig, ax = plt.subplots(figsize=(11, 4.5))
                bar_colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in ind_returns["Annualised Return (%)"]]
                ax.bar(ind_returns["Stock"], ind_returns["Annualised Return (%)"], color=bar_colors)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title("Annualised Returns per Stock (CAGR %)", fontsize=13, fontweight="bold")
                ax.set_ylabel("Return (%)")
                ax.grid(alpha=0.3, axis="y")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with tab4:
            sector_df = pd.DataFrame({"Sector": valid_sectors, "Weight": valid_weights})
            sector_agg = sector_df.groupby("Sector")["Weight"].sum()

            fig, ax = plt.subplots(figsize=(6, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(sector_agg)))
            ax.pie(sector_agg, labels=sector_agg.index, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.set_title("Portfolio Sector Allocation", fontsize=13, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Portfolio Optimization Workflow
    if optimize_clicked:
        optimize_portfolio(stock_data[valid_tickers], n_scenarios)


if __name__ == "__main__":
    main()
