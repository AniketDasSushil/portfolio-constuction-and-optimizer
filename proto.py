"""
Portfolio Constructor, Analyzer and Optimizer
ULTRA-OPTIMIZED for Streamlit 1.50+ with session state caching.

Key improvements:
1. Removed ThreadPoolExecutor (causes thread leaks in Streamlit)
2. Added session state caching (prevents redownloading)
3. Simplified progress (single progress bar, no nested UI)
4. Used @st.cache_resource for yfinance session
5. All previous bug fixes included
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time
import logging

# Suppress warnings and debug output
logging.getLogger("yfinance").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for caching
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = None
if 'market_data_cache' not in st.session_state:
    st.session_state.market_data_cache = None
if 'cache_params' not in st.session_state:
    st.session_state.cache_params = None


# ─────────────────────────────────────────────────────────────────────────
# YFINANCE SESSION (Persistent across reruns)
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_yfinance_session():
    """Create a persistent yfinance session with caching."""
    try:
        from requests_cache import CachedSession
        return CachedSession(
            'yfinance_cache',
            expire_after=3600,  # Cache for 1 hour
            stale_if_error=True
        )
    except ImportError:
        # Fallback if requests_cache not installed
        import requests
        return requests.Session()


# ─────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Simplified, no threading)
# ─────────────────────────────────────────────────────────────────────────

def fetch_stock_data(tickers: tuple, years: int) -> pd.DataFrame:
    """
    Download stock data with smart caching.
    Only downloads if not already in session state.
    """
    # Check if we have cached data for these exact parameters
    cache_key = (tickers, years)
    
    if (st.session_state.stock_data_cache is not None and 
        st.session_state.cache_params == cache_key):
        st.info("✅ Using cached stock data")
        return st.session_state.stock_data_cache
    
    session = get_yfinance_session()
    st.write("📊 **Downloading stock prices...**")
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    data_dict = {}
    errors = []
    
    for i, ticker in enumerate(tickers):
        try:
            # Update progress
            progress = (i + 1) / len(tickers)
            progress_bar.progress(progress)
            status_text.text(f"⏳ {ticker} ({i+1}/{len(tickers)})")
            
            # Download with timeout
            raw = yf.download(
                ticker,
                period=f"{years}y",
                auto_adjust=True,
                progress=False,
                timeout=30,
                session=session
            )
            
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                if "Close" in raw.columns:
                    data_dict[ticker] = raw["Close"]
                    status_text.text(f"✅ {ticker} ({i+1}/{len(tickers)})")
            else:
                errors.append(ticker)
                status_text.text(f"❌ {ticker} - No data ({i+1}/{len(tickers)})")
                
        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:40]}")
            status_text.text(f"❌ {ticker} - Error ({i+1}/{len(tickers)})")
    
    # Clean up progress bar
    progress_bar.empty()
    status_text.empty()
    
    if errors:
        st.warning(f"⚠️ Failed to download: {', '.join(errors[:3])}")
    
    if not data_dict:
        st.error("❌ No data downloaded. Check tickers and internet.")
        return pd.DataFrame()
    
    # Combine and clean
    data = pd.DataFrame(data_dict)
    data = data.dropna(axis=1, how="all")
    data = data.ffill().dropna()
    
    # Cache in session state
    st.session_state.stock_data_cache = data
    st.session_state.cache_params = cache_key
    
    return data


def fetch_market_data(years: int) -> pd.Series:
    """Download NIFTY 50 with session caching."""
    if (st.session_state.market_data_cache is not None and
        st.session_state.cache_params is not None):
        return st.session_state.market_data_cache
    
    session = get_yfinance_session()
    
    try:
        raw = yf.download(
            "^NSEI",
            period=f"{years}y",
            auto_adjust=True,
            progress=False,
            timeout=30,
            session=session
        )
        
        if isinstance(raw, pd.DataFrame):
            series = raw["Close"].squeeze() if "Close" in raw.columns else pd.Series()
        else:
            series = raw
        
        series = series.ffill().dropna()
        st.session_state.market_data_cache = series
        return series
        
    except Exception as e:
        st.error(f"❌ Could not download NIFTY 50: {str(e)[:100]}")
        return pd.Series()


# ─────────────────────────────────────────────────────────────────────────
# CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def portfolio_create(
    tickers: tuple, weights: tuple, data: pd.DataFrame
) -> pd.Series:
    """Calculate portfolio cumulative returns."""
    weights_arr = np.array(weights, dtype=float)
    valid_tickers = [t for t in tickers if t in data.columns]
    
    if not valid_tickers:
        return pd.Series()
    
    ticker_to_weight = {tickers[i]: weights_arr[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()

    ret = data[valid_tickers].pct_change().dropna()
    if ret.empty:
        return pd.Series()
    
    portfolio_returns = (ret * valid_weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()
    cum_returns = cum_returns / cum_returns.iloc[0]
    
    return cum_returns.rename("Portfolio")


@st.cache_data
def calculate_cumulative_returns(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Calculate cumulative returns."""
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
    """Calculate portfolio metrics."""
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
    beta = covariance / market_variance if market_variance > 0 else 0

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
    """Calculate individual stock returns."""
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
# OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────

@st.cache_data
def run_optimization(stock_data_cols: tuple, stock_data_values: tuple, n_scenarios: int) -> dict:
    """Run Monte Carlo optimization (cached)."""
    # Reconstruct DataFrame from cache-safe tuples
    stock_data = pd.DataFrame(
        {col: np.array(vals) for col, vals in zip(stock_data_cols, stock_data_values)}
    )
    
    actual_tickers = list(stock_data.columns)
    returns = stock_data.pct_change().dropna()

    results = {"weights": [], "returns": [], "risks": [], "sharpe": []}

    for _ in range(n_scenarios):
        w = np.random.random(len(actual_tickers))
        w /= w.sum()

        port_return = (returns.mean() * w).sum() * 252
        port_risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        sharpe = port_return / port_risk if port_risk > 0 else 0.0

        results["weights"].append(w)
        results["returns"].append(port_return)
        results["risks"].append(port_risk)
        results["sharpe"].append(sharpe)

    optimal_idx = int(np.argmax(results["sharpe"]))
    
    return {
        "tickers": actual_tickers,
        "optimal_idx": optimal_idx,
        "results": results,
    }


def optimize_portfolio(stock_data: pd.DataFrame, tickers: list[str], n_scenarios: int) -> None:
    """Run optimization with progress."""
    actual_tickers = list(stock_data.columns)
    
    # Convert to hashable tuples for caching
    cols_tuple = tuple(actual_tickers)
    vals_tuple = tuple(stock_data[col].values for col in actual_tickers)
    
    st.write(f"🎯 Running {n_scenarios:,} simulations...")
    
    opt_result = run_optimization(cols_tuple, vals_tuple, n_scenarios)
    results = opt_result["results"]
    optimal_idx = opt_result["optimal_idx"]
    
    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(
        results["risks"],
        results["returns"],
        c=results["sharpe"],
        cmap="plasma",
        alpha=0.6,
        s=15,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(
        results["risks"][optimal_idx],
        results["returns"][optimal_idx],
        color="red",
        marker="*",
        s=700,
        label="Optimal",
        zorder=5,
        edgecolors="darkred",
        linewidth=2
    )
    ax.set_xlabel("Risk (Volatility)", fontsize=12)
    ax.set_ylabel("Return (CAGR %)", fontsize=12)
    ax.set_title("Efficient Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Allocation table
    st.success("✅ Optimal Portfolio (Maximum Sharpe Ratio)")
    optimal_df = pd.DataFrame({
        "Stock": actual_tickers,
        "Weight (%)": np.round(results["weights"][optimal_idx] * 100, 2),
    }).sort_values("Weight (%)", ascending=False)
    st.dataframe(optimal_df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", f"{results['sharpe'][optimal_idx]:.4f}")
    col2.metric("Expected Return", f"{results['returns'][optimal_idx]*100:.2f}%")
    col3.metric("Risk (Volatility)", f"{results['risks'][optimal_idx]*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("🏦 Portfolio Constructor & Optimizer")
    st.write("*Fast, reliable portfolio analysis using cached data*")

    # Load symbols
    try:
        sym = pd.read_csv("sym.csv")
        sym.columns = [col.replace(" ", "_") for col in sym.columns]
    except FileNotFoundError:
        st.error("❌ 'sym.csv' not found. Place it in the working directory.")
        return

    required_cols = {"NAME_OF_COMPANY", "SYMBOL", "SECTOR"}
    if not required_cols.issubset(sym.columns):
        st.error(f"❌ sym.csv missing: {required_cols}")
        return

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        n_stocks = st.slider("Number of stocks", 2, 20, 5)
        years = st.slider("Data period (years)", 1, 20, 5)
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (%)", value=6.66, min_value=0.0, max_value=20.0, step=0.1
        )
        n_scenarios = st.slider("Optimization scenarios", 500, 5000, 1000, step=500)
        
        st.divider()
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.session_state.stock_data_cache = None
            st.session_state.market_data_cache = None
            st.session_state.cache_params = None
            st.rerun()

    # Stock selection
    st.subheader("📈 Build Portfolio")
    tickers = []
    weights = []
    sectors = []

    cols = st.columns(n_stocks)
    for i, col in enumerate(cols):
        with col:
            name = st.selectbox(
                f"Stock {i+1}",
                sym["NAME_OF_COMPANY"],
                key=f"stock_{i}",
                label_visibility="collapsed"
            )
            row = sym[sym["NAME_OF_COMPANY"] == name].iloc[0]
            tickers.append(f"{row['SYMBOL']}.NS")
            sectors.append(row["SECTOR"])

    weight_cols = st.columns(n_stocks)
    for i, col in enumerate(weight_cols):
        with col:
            w = st.number_input(
                f"Weight",
                min_value=0.0,
                max_value=100.0,
                value=round(100 / n_stocks, 1),
                step=1.0,
                key=f"weight_{i}",
                label_visibility="collapsed"
            )
            weights.append(w)

    # Validation
    unique_tickers = set(tickers)
    if len(unique_tickers) < len(tickers):
        duplicates = list(set([t for t in tickers if tickers.count(t) > 1]))
        st.error(f"❌ Duplicate stocks: {duplicates}")
        st.stop()

    total_weight = sum(weights)
    weight_ok = abs(total_weight - 100.0) < 1e-6

    if not weight_ok:
        st.error(f"⚠️ Total weight is {total_weight:.1f}% (need 100%)")
        st.stop()

    # Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    analyze_button = col1.button("📊 Analyse", use_container_width=True)
    optimize_button = col2.button("🎯 Optimize", use_container_width=True)

    if not (analyze_button or optimize_button):
        return

    # Download data
    tickers_tuple = tuple(tickers)
    weights_tuple = tuple(np.array(weights, dtype=float) / 100.0)

    stock_data = fetch_stock_data(tickers_tuple, years)
    market_series = fetch_market_data(years)

    if stock_data.empty or market_series.empty:
        st.error("❌ Could not download data. Try again or check tickers.")
        st.stop()

    valid_tickers = [t for t in tickers if t in stock_data.columns]
    if len(valid_tickers) < 2:
        st.error("❌ Less than 2 valid stocks.")
        st.stop()

    ticker_to_weight_orig = {tickers[i]: weights_tuple[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight_orig[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()

    ticker_to_sector = {tickers[i]: sectors[i] for i in range(len(tickers))}
    valid_sectors = [ticker_to_sector[t] for t in valid_tickers]

    # ANALYSE
    if analyze_button:
        st.info("📈 Calculating metrics...")
        
        portfolio_cum = portfolio_create(tickers_tuple, weights_tuple, stock_data)
        if portfolio_cum.empty:
            st.error("❌ Could not create portfolio.")
            st.stop()

        tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Metrics", "Stocks", "Allocations"])

        with tab1:
            # Portfolio vs Market
            market_cum = calculate_cumulative_returns(market_series).rename("NIFTY 50")
            compare_df = pd.concat([portfolio_cum, market_cum], axis=1).dropna()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            compare_df.plot(ax=ax, linewidth=2.5)
            ax.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
            ax.set_ylabel("Growth of ₹1")
            ax.axhline(1, color="grey", linestyle="--", alpha=0.5)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=11)
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
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" 
                         for v in ind_returns["Annualised Return (%)"]]
                ax.bar(ind_returns["Stock"], ind_returns["Annualised Return (%)"], color=colors)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title("Annualised Returns by Stock", fontsize=14, fontweight="bold")
                ax.set_ylabel("Return (%)")
                ax.grid(alpha=0.3, axis="y")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with tab4:
            sector_df = pd.DataFrame({"Sector": valid_sectors, "Weight": valid_weights})
            sector_agg = sector_df.groupby("Sector")["Weight"].sum()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.Set3(np.linspace(0, 1, len(sector_agg)))
            ax.pie(sector_agg, labels=sector_agg.index, autopct="%1.1f%%",
                   colors=colors, startangle=90)
            ax.set_title("Sector Allocation", fontsize=14, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # OPTIMIZE
    if optimize_button:
        optimize_portfolio(stock_data[valid_tickers], valid_tickers, n_scenarios)


if __name__ == "__main__":
    main()
