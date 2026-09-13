"""
Portfolio Constructor, Analyzer and Optimizer
Fixed version with all bugs resolved.

Critical fixes:
1. Fixed weight-to-ticker mapping using explicit index dictionaries
2. Fixed sector allocation indexing
3. Fixed Sharpe ratio calculation (use excess_returns.std())
4. Fixed market data type handling
5. Fixed optimization loop to use actual columns
6. Added duplicate ticker detection
7. Added control flow stop() after validation errors
8. Fixed cache inconsistency for filtered portfolios
9. Added handling for extreme negative returns
10. Improved data alignment across functions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_stock_data(tickers: tuple, years: int) -> pd.DataFrame:
    """Download and clean adjusted close prices for given tickers."""
    raw = yf.download(list(tickers), period=f"{years}y", auto_adjust=True, progress=False)

    # auto_adjust=True gives 'Close' instead of 'Adj Close'
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"]
    else:
        # Single ticker — flat DataFrame; rename column to the ticker symbol
        data = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Drop columns that are entirely NaN (delisted / bad tickers)
    bad = data.columns[data.isna().all()].tolist()
    if bad:
        st.warning(f"No data returned for: {bad}. They will be excluded.")
    data = data.dropna(axis=1, how="all")

    # Forward-fill then drop any remaining NaNs at the start
    data = data.ffill().dropna()
    return data


@st.cache_data
def fetch_market_data(years: int) -> pd.Series:
    """Download NIFTY 50 index data and return as a Series."""
    raw = yf.download("^NSEI", period=f"{years}y", auto_adjust=True, progress=False)
    
    # Handle both DataFrame and Series returns
    if isinstance(raw, pd.DataFrame):
        if isinstance(raw.columns, pd.MultiIndex):
            series = raw["Close"].squeeze()
        else:
            series = raw["Close"].squeeze()
    else:
        series = raw
    
    return series.ffill().dropna()


# ---------------------------------------------------------------------------
# Calculation functions (cache-safe: use only hashable primitive types)
# ---------------------------------------------------------------------------

@st.cache_data
def portfolio_create(
    tickers: tuple, weights: tuple, data: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio cumulative returns.
    Weights must be provided as a tuple so Streamlit can hash them.
    Cumulative returns are normalised to start at 1.0.
    
    FIX: Properly aligns weights with filtered tickers using index mapping.
    """
    weights_arr = np.array(weights)
    # Keep only columns present in data (some tickers may have been dropped)
    valid_tickers = [t for t in tickers if t in data.columns]
    
    # FIX: Use tuple indexing to map original positions to weights
    ticker_to_weight = {tickers[i]: weights[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()  # Re-normalise after possible exclusions

    ret = data[valid_tickers].pct_change().dropna()
    portfolio_returns = (ret * valid_weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()

    # Normalise so the series starts exactly at 1.0
    cum_returns = cum_returns / cum_returns.iloc[0]
    return cum_returns.rename("Portfolio")


@st.cache_data
def calculate_cumulative_returns(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Calculate cumulative returns normalised to start at 1.0."""
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
    risk_free_rate_pct: float = 6.66,   # ← in PERCENT (e.g. 6.66 means 6.66 %)
) -> pd.DataFrame:
    """
    Calculate portfolio metrics including CAPM, beta, alpha, Sharpe ratio.

    All return/rate values are in PERCENT for consistency.
    risk_free_rate_pct : annual risk-free rate expressed as a percentage (e.g. 6.66).
    
    FIX: Corrected Sharpe ratio calculation to use excess_returns.std().
    """
    portfolio_returns = portfolio_cum.pct_change().dropna()
    market_returns = market_series.pct_change().dropna()

    # Align on common dates
    portfolio_returns, market_returns = portfolio_returns.align(
        market_returns, join="inner"
    )

    if portfolio_returns.empty or market_returns.empty:
        st.error("Could not align portfolio and market return series.")
        return pd.DataFrame()

    # Beta
    covariance = portfolio_returns.cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance

    # CAGR values — expressed as percentages
    market_start = market_series.iloc[0]
    market_end = market_series.iloc[-1]
    market_cagr_pct = (((market_end / market_start) ** (1 / years)) - 1) * 100

    portfolio_start = portfolio_cum.iloc[0]
    portfolio_end = portfolio_cum.iloc[-1]
    portfolio_cagr_pct = (((portfolio_end / portfolio_start) ** (1 / years)) - 1) * 100

    # CAPM expected return — all values in PERCENT
    capm_return_pct = risk_free_rate_pct + beta * (market_cagr_pct - risk_free_rate_pct)

    # Annualised volatility (expressed as %)
    portfolio_volatility_pct = portfolio_returns.std() * np.sqrt(252) * 100

    # FIX: Sharpe ratio now correctly uses excess_returns.std() in denominator
    daily_rf = (1 + risk_free_rate_pct / 100) ** (1 / 252) - 1
    excess_returns = portfolio_returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0.0

    # Correlation with market
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
    """
    Calculate annualised returns and volatility for individual stocks.
    
    FIX: Added handling for extreme negative returns (negative total_return > -100%).
    """
    rows = []
    for col in stock_data.columns:
        series = stock_data[col].dropna()
        if len(series) < 2:
            continue
        
        # FIX: Handle edge case where stock loses 100%+ of value
        total_return = (series.iloc[-1] / series.iloc[0]) - 1
        
        # If loss is greater than 100%, annualized return is undefined
        if 1 + total_return <= 0:
            ann_return_pct = -100.0  # Stock completely lost value
        else:
            total_return_pct = total_return * 100
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

    df = pd.DataFrame(rows)
    return df.sort_values("Annualised Return (%)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def optimize_portfolio(
    stock_data: pd.DataFrame, tickers: list[str], n_scenarios: int
) -> None:
    """
    Run Markowitz-style Monte Carlo portfolio optimisation.
    
    FIX: Uses stock_data.columns instead of tickers to ensure alignment.
    """
    # FIX: Use actual columns from data, not input tickers
    actual_tickers = list(stock_data.columns)
    returns = stock_data.pct_change().dropna()

    results: dict[str, list] = {"weights": [], "returns": [], "risks": [], "sharpe": []}

    with st.spinner("Running optimisation…"):
        for _ in range(n_scenarios):
            w = np.random.random(len(actual_tickers))  # FIX: Use len(actual_tickers)
            w /= w.sum()

            port_return = (returns.mean() * w).sum() * 252
            port_risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
            sharpe = port_return / port_risk if port_risk > 0 else 0.0

            results["weights"].append(w)
            results["returns"].append(port_return)
            results["risks"].append(port_risk)
            results["sharpe"].append(sharpe)

    optimal_idx = int(np.argmax(results["sharpe"]))

    # Efficient frontier plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        results["risks"],
        results["returns"],
        c=results["sharpe"],
        cmap="plasma",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.scatter(
        results["risks"][optimal_idx],
        results["returns"][optimal_idx],
        color="red",
        marker="*",
        s=500,
        label="Optimal Portfolio",
        zorder=5,
    )
    ax.set_xlabel("Risk (Annualised Volatility)")
    ax.set_ylabel("Expected Return (Annualised)")
    ax.set_title("Efficient Frontier")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Optimal allocation table
    st.success("Optimal Portfolio Allocation (Maximum Sharpe Ratio):")
    optimal_df = pd.DataFrame(
        {
            "Stock": actual_tickers,
            "Weight (%)": np.round(results["weights"][optimal_idx] * 100, 2),
        }
    ).sort_values("Weight (%)", ascending=False)
    st.table(optimal_df)

    best_sharpe = results["sharpe"][optimal_idx]
    best_return = results["returns"][optimal_idx] * 100
    best_risk = results["risks"][optimal_idx] * 100
    st.info(
        f"**Optimal Sharpe**: {best_sharpe:.4f} | "
        f"**Return**: {best_return:.2f}% | "
        f"**Volatility**: {best_risk:.2f}%"
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Portfolio Constructor, Analyser and Optimiser")

    # ── Load symbol CSV ──────────────────────────────────────────────────────
    try:
        sym = pd.read_csv("sym.csv")
        sym.columns = [col.replace(" ", "_") for col in sym.columns]
    except FileNotFoundError:
        st.error("Symbol data file 'sym.csv' not found in the working directory.")
        return

    required_cols = {"NAME_OF_COMPANY", "SYMBOL", "SECTOR"}
    if not required_cols.issubset(sym.columns):
        st.error(f"sym.csv must contain columns: {required_cols}. Found: {set(sym.columns)}")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        n_stocks = st.slider("Number of stocks", 2, 20, 5)
        years = st.slider("Look-back period (years)", 1, 20, 5)
        risk_free_rate_pct = st.number_input(
            "Risk-free rate (%)", value=6.66, min_value=0.0, max_value=20.0, step=0.1,
            help="Annual risk-free rate expressed as a percentage (e.g. 6.66 for 6.66%)."
        )
        n_scenarios = st.slider("Optimisation scenarios", 500, 5000, 1000, step=500)

    # ── Stock selection & weights ────────────────────────────────────────────
    st.subheader("Portfolio Construction")
    tickers: list[str] = []
    weights: list[float] = []
    sectors: list[str] = []

    for i in range(n_stocks):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.selectbox(
                f"Stock #{i + 1}", sym["NAME_OF_COMPANY"], key=f"stock_{i}"
            )
            row = sym[sym["NAME_OF_COMPANY"] == name].iloc[0]
            tickers.append(f"{row['SYMBOL']}.NS")
            sectors.append(row["SECTOR"])
        with col2:
            w = st.number_input(
                "Weight (%)", key=f"weight_{i}",
                min_value=0.0, max_value=100.0, value=round(100 / n_stocks, 1), step=5.0
            )
            weights.append(w)

    # FIX: Detect duplicate tickers early
    unique_tickers = set(tickers)
    if len(unique_tickers) < len(tickers):
        duplicates = [t for t in tickers if tickers.count(t) > 1]
        st.error(f"Duplicate stocks selected: {list(set(duplicates))}. Please select unique stocks.")
        st.stop()  # FIX: Stop execution here

    total_weight = sum(weights)
    weight_ok = abs(total_weight - 100.0) < 1e-6

    if not weight_ok:
        st.warning(f"Total weight is {total_weight:.1f}% — must equal 100% before proceeding.")

    # ── Action buttons ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    analyze_button = col1.button("Analyse Portfolio", disabled=not weight_ok)
    optimize_button = col2.button("Optimise Portfolio", disabled=not weight_ok)

    if not (analyze_button or optimize_button):
        return

    # ── Normalise weights and convert to tuple for cache-safe hashing ────────
    weights_arr = np.array(weights, dtype=float) / 100.0
    tickers_tuple = tuple(tickers)
    weights_tuple = tuple(weights_arr)

    # ── Download data ────────────────────────────────────────────────────────
    with st.spinner("Downloading price data…"):
        stock_data = fetch_stock_data(tickers_tuple, years)
        market_series = fetch_market_data(years)

    if stock_data.empty:
        st.error("No stock data could be downloaded. Check ticker symbols.")
        st.stop()  # FIX: Stop execution here

    if market_series.empty:
        st.error("Could not download NIFTY 50 benchmark data.")
        st.stop()  # FIX: Stop execution here

    # Update tickers/weights to only valid (downloaded) stocks
    valid_tickers = [t for t in tickers if t in stock_data.columns]
    if len(valid_tickers) < 2:
        st.error("Fewer than 2 valid tickers — cannot build a portfolio.")
        st.stop()  # FIX: Stop execution here

    # FIX: Proper weight mapping using dictionary
    ticker_to_weight_orig = {tickers[i]: weights_arr[i] for i in range(len(tickers))}
    valid_weights = np.array([ticker_to_weight_orig[t] for t in valid_tickers])
    valid_weights /= valid_weights.sum()

    # FIX: Proper sector mapping using dictionary
    ticker_to_sector = {tickers[i]: sectors[i] for i in range(len(tickers))}
    valid_sectors = [ticker_to_sector[t] for t in valid_tickers]

    # ── ANALYSE ──────────────────────────────────────────────────────────────
    if analyze_button:
        portfolio_cum = portfolio_create(tickers_tuple, weights_tuple, stock_data)

        # 1. Portfolio cumulative returns
        st.subheader("Portfolio Cumulative Returns")
        fig, ax = plt.subplots(figsize=(10, 5))
        portfolio_cum.plot(ax=ax, color="steelblue", linewidth=2)
        ax.set_title("Portfolio Cumulative Returns")
        ax.set_ylabel("Growth of ₹1")
        ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        st.pyplot(fig)
        plt.close(fig)

        # 2. Market comparison
        st.subheader("Portfolio vs NIFTY 50")
        market_cum = calculate_cumulative_returns(market_series).rename("NIFTY 50")
        compare_df = pd.concat([portfolio_cum, market_cum], axis=1).dropna()
        fig, ax = plt.subplots(figsize=(10, 5))
        compare_df.plot(ax=ax)
        ax.set_title("Cumulative Returns: Portfolio vs NIFTY 50")
        ax.set_ylabel("Growth of ₹1")
        ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        st.pyplot(fig)
        plt.close(fig)

        # 3. Portfolio composition (pie chart — FIX: proper sector mapping)
        st.subheader("Sector Allocation")
        sector_df = pd.DataFrame({"Sector": valid_sectors, "Weight": valid_weights})
        sector_agg = sector_df.groupby("Sector")["Weight"].sum()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(sector_agg, labels=sector_agg.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Portfolio Sector Weights")
        st.pyplot(fig)
        plt.close(fig)

        # 4. Correlation heatmap
        st.subheader("Stock Correlations")
        fig, ax = plt.subplots(figsize=(max(6, len(valid_tickers)), max(5, len(valid_tickers) - 1)))
        corr_matrix = stock_data[valid_tickers].pct_change().dropna().corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    vmin=-1, vmax=1, center=0)
        ax.set_title("Pairwise Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)

        # 5. Portfolio metrics
        st.subheader("Portfolio Metrics")
        metrics_df = calculate_metrics(
            portfolio_cum, market_series, years, risk_free_rate_pct
        )
        if not metrics_df.empty:
            st.table(metrics_df)

        # 6. Individual stock performance
        st.subheader("Individual Stock Performance")
        ind_returns = calculate_individual_returns(stock_data[valid_tickers], years)
        st.table(ind_returns)

        # Bar chart — annualised returns
        fig, ax = plt.subplots(figsize=(max(8, len(valid_tickers) * 1.2), 5))
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ind_returns["Annualised Return (%)"]]
        bars = ax.bar(ind_returns["Stock"], ind_returns["Annualised Return (%)"], color=colors)
        ax.set_title("Annualised Returns by Stock")
        ax.set_xlabel("Stock")
        ax.set_ylabel("Annualised Return (%)")
        ax.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.3 if height >= 0 else -1.0),
                f"{height:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Scatter — risk vs return
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            ind_returns["Volatility (%)"],
            ind_returns["Annualised Return (%)"],
            color="steelblue", s=80, zorder=3,
        )
        for _, row in ind_returns.iterrows():
            ax.annotate(
                row["Stock"],
                (row["Volatility (%)"], row["Annualised Return (%)"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title("Risk–Return Profile of Individual Stocks")
        ax.set_xlabel("Annualised Volatility (%)")
        ax.set_ylabel("Annualised Return (%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 7. Cumulative returns — individual stocks
        st.subheader("Cumulative Returns Over Time")
        cum_individual = calculate_cumulative_returns(stock_data[valid_tickers])
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in cum_individual.columns:
            ax.plot(cum_individual.index, cum_individual[col], label=col, linewidth=1.5)
        ax.axhline(1, color="grey", linestyle="--", linewidth=0.8)
        ax.set_title("Cumulative Returns of Individual Stocks")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of ₹1")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── OPTIMISE ──────────────────────────────────────────────────────────────
    if optimize_button:
        st.subheader("Portfolio Optimisation (Markowitz / Max Sharpe)")
        optimize_portfolio(stock_data[valid_tickers], valid_tickers, n_scenarios)


if __name__ == "__main__":
    main()
