import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Configuring Caching Functions
@st.cache_data
def fetch_stock_data(tickers, years):
    return yf.download(tickers, period=f'{years}y')['Adj Close']

@st.cache_data
def compute_cumulative_returns(data):
    returns = data.pct_change()
    cumulative = (1 + returns).cumprod()
    return cumulative

@st.cache_data
def calculate_portfolio_metrics(portfolio_data, market_data, years):
    ret = portfolio_data.pct_change()
    market_ret = market_data.pct_change()

    # Portfolio Metrics
    mean_ret = ret.mean() * 252
    vol = ret.std() * np.sqrt(252)
    sharpe_ratio = mean_ret / vol

    # Beta Calculation
    cov = ret.cov()
    beta = cov.iloc[0, 1] / market_ret.var()

    # CAPM and Alpha
    market_return = ((market_data.iloc[-1] / market_data.iloc[0]) ** (1 / years) - 1) * 100
    risk_free_rate = 6.66
    capm_return = risk_free_rate + beta * (market_return - risk_free_rate)
    alpha = mean_ret * 100 - capm_return

    return {
        "Mean Return": mean_ret,
        "Volatility": vol,
        "Sharpe Ratio": sharpe_ratio,
        "Beta": beta,
        "CAPM Return": capm_return,
        "Alpha": alpha
    }

# Streamlit App Title
st.title('Portfolio Constructor, Analyzer, and Optimizer')

# Load Symbol Data
sym = pd.read_csv('sym.csv')
sym.columns = [column.replace(' ', '_') for column in sym.columns]

# User Inputs
n = st.slider('Select the number of stocks in your portfolio', min_value=2, max_value=20, step=1)
years = st.slider('Select the number of years for analysis', min_value=1, max_value=20, step=1)

# Initialize Variables
tickers = []
weights = []
sectors = []

# Stock Selection
for i in range(n):
    stock_name = st.selectbox(f'Select Stock {i + 1}', sym['NAME_OF_COMPANY'], key=f'stock_{i}')
    weight = st.number_input(f'Enter Weight for {stock_name} (%)', min_value=0, max_value=100, step=1, key=f'weight_{i}')

    # Fetch ticker and sector
    selected_stock = sym[sym['NAME_OF_COMPANY'] == stock_name]
    tickers.append(selected_stock['SYMBOL'].iloc[0] + ".NS")
    weights.append(weight)
    sectors.append(selected_stock['SECTOR'].iloc[0])

# Check Total Weight
if sum(weights) != 100:
    st.warning(f'Total weights must sum to 100%. Current total: {sum(weights)}%.')

# Confirm Button
if st.button('Confirm Portfolio'):
    weights = np.array(weights) / 100  # Normalize weights

    # Fetch Data
    stock_data = fetch_stock_data(tickers, years)
    market_data = fetch_stock_data(['^NSEI'], years)

    # Calculate Portfolio Returns
    portfolio_returns = (stock_data.pct_change() * weights).sum(axis=1)
    cumulative_portfolio = (1 + portfolio_returns).cumprod()

    # Plot Cumulative Returns
    fig, ax = plt.subplots()
    ax.plot(cumulative_portfolio, label='Portfolio')
    ax.set_title('Cumulative Returns of Your Portfolio')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    st.pyplot(fig)

    # Compare with Market
    compare = st.radio('Compare Portfolio with Market?', ['No', 'Yes'])
    if compare == 'Yes':
        cumulative_market = compute_cumulative_returns(market_data)
        cumulative_market.rename(columns={'Adj Close': 'Market'}, inplace=True)
        comparison = pd.concat([cumulative_portfolio, cumulative_market], axis=1)

        fig, ax = plt.subplots()
        comparison.plot(ax=ax)
        ax.set_title('Portfolio vs Market Cumulative Returns')
        ax.legend(['Portfolio', 'Market'])
        st.pyplot(fig)

    # Portfolio Composition
    fig, ax = plt.subplots()
    ax.pie(weights, labels=sectors, autopct='%1.1f%%')
    ax.set_title('Portfolio Composition by Sector')
    st.pyplot(fig)

    # Metrics Calculation
    metrics = calculate_portfolio_metrics(cumulative_portfolio, market_data, years)
    st.subheader('Portfolio Metrics')
    st.table(metrics)

# Portfolio Optimization (Markowitz Model)
scenarios = st.slider('Select the Number of Scenarios for Optimization', min_value=500, max_value=5000, step=500)

if st.button('Start Optimization'):
    returns = stock_data.pct_change()
    p_weights = []
    p_returns = []
    p_risks = []
    p_sharpe = []

    for _ in range(scenarios):
        wts = np.random.random(len(tickers))
        wts /= np.sum(wts)

        # Calculate Portfolio Return and Risk
        annual_return = np.sum(returns.mean() * wts) * 252
        portfolio_std = np.sqrt(np.dot(wts.T, np.dot(returns.cov() * 252, wts)))

        # Sharpe Ratio
        sharpe_ratio = annual_return / portfolio_std

        p_weights.append(wts)
        p_returns.append(annual_return)
        p_risks.append(portfolio_std)
        p_sharpe.append(sharpe_ratio)

    # Find Optimal Portfolio
    max_sharpe_idx = np.argmax(p_sharpe)
    optimal_weights = p_weights[max_sharpe_idx]

    # Plot Efficient Frontier
    fig, ax = plt.subplots()
    scatter = ax.scatter(p_risks, p_returns, c=p_sharpe, cmap='viridis')
    ax.scatter(p_risks[max_sharpe_idx], p_returns[max_sharpe_idx], color='r', marker='*', s=200, label='Optimal Portfolio')
    ax.set_title('Efficient Frontier')
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Return')
    ax.legend()
    st.pyplot(fig)

    st.success(f'Optimal Weights: {dict(zip(tickers, np.round(optimal_weights, 2)))}')
