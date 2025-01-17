import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Cache decorators should use hash_funcs for mutable data
@st.cache_data
def portfolio_create(tickers, weights, data):
    """Calculate portfolio cumulative returns"""
    ret = data.pct_change()
    portfolio_returns = (ret * weights).sum(axis=1)
    cum_returns = (1 + portfolio_returns).cumprod()
    return cum_returns.rename('Portfolio')

@st.cache_data
def calculate_cumulative_returns(data):
    """Calculate cumulative returns for given data"""
    ret = data.pct_change()
    return (1 + ret).cumprod()

@st.cache_data
def calculate_metrics(portfolio_data, market_data, risk_free_rate=0.0666):
    """Calculate portfolio metrics including CAPM, beta, alpha, etc."""
    # Calculate returns
    portfolio_returns = portfolio_data.pct_change()
    market_returns = market_data.pct_change()
    
    # Calculate beta
    covariance = portfolio_returns['Portfolio'].cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance
    
    # Calculate CAPM expected return
    years = (portfolio_data.index[-1] - portfolio_data.index[0]).days / 365
    market_cagr = (((market_data[-1]/market_data[0])**(1/years))-1)*100
    capm_return = risk_free_rate + beta * (market_cagr - risk_free_rate)
    
    # Calculate actual portfolio performance
    portfolio_cagr = (((portfolio_data[-1]/portfolio_data[0])**(1/years))-1)*100
    portfolio_volatility = portfolio_returns['Portfolio'].std() * np.sqrt(252)
    correlation = portfolio_returns['Portfolio'].corr(market_returns)
    
    metrics = {
        'Beta': beta,
        'CAPM Expected Return (%)': capm_return,
        'Actual Return (%)': portfolio_cagr,
        'Volatility (%)': portfolio_volatility * 100,
        'Alpha (%)': portfolio_cagr - capm_return,
        'Correlation': correlation
    }
    
    return pd.DataFrame(metrics, index=['Portfolio']).round(4)

def main():
    st.title('Portfolio Constructor, Analyzer and Optimizer')
    
    # Load symbol data
    try:
        sym = pd.read_csv('sym.csv')
        sym.columns = [col.replace(" ", '_') for col in sym.columns]
    except FileNotFoundError:
        st.error("Symbol data file 'sym.csv' not found!")
        return
    
    # Portfolio construction inputs
    n_stocks = st.slider('Select number of stocks', 2, 20)
    years = st.slider('Number of years', 1, 20)
    
    # Stock selection and weight input
    tickers = []
    weights = []
    sectors = []
    
    for i in range(n_stocks):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.selectbox(f'Select stock #{i+1}', sym['NAME_OF_COMPANY'], key=f'stock_{i}')
            stock_data = sym[sym['NAME_OF_COMPANY'] == name].iloc[0]
            tickers.append(f"{stock_data['SYMBOL']}.NS")
            sectors.append(stock_data['SECTOR'])
        
        with col2:
            weight = st.number_input(f'Weight (%)', key=f'weight_{i}', min_value=0.0, max_value=100.0, step=5.0)
            weights.append(weight)
    
    total_weight = sum(weights)
    if total_weight != 100:
        st.warning(f'Total weight is {total_weight}%. It should be 100%')
    
    if st.button('Analyze Portfolio'):
        weights = np.array(weights) / 100  # Normalize weights
        
        # Download data
        with st.spinner('Downloading data...'):
            stock_data = yf.download(tickers, period=f'{years}y')['Adj Close']
            market_data = yf.download('^NSEI', period=f'{years}y')['Adj Close']
            
            if stock_data.empty or market_data.empty:
                st.error('Failed to download data!')
                return
        
        # Calculate portfolio performance
        portfolio_data = portfolio_create(tickers, weights, stock_data)
        
        # Display visualizations
        st.subheader('Portfolio Performance')
        fig, ax = plt.subplots(figsize=(10, 6))
        portfolio_data.plot(ax=ax)
        ax.set_title('Cumulative Returns')
        st.pyplot(fig)
        
        # Market comparison
        st.subheader('Market Comparison')
        fig, ax = plt.subplots(figsize=(10, 6))
        compare_data = pd.concat([
            calculate_cumulative_returns(market_data).rename('NIFTY'),
            portfolio_data
        ], axis=1)
        compare_data.plot(ax=ax)
        st.pyplot(fig)
        
        # Portfolio composition
        st.subheader('Portfolio Composition')
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.pie(weights, labels=sectors, autopct='%1.1f%%')
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader('Stock Correlations')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(stock_data.corr(), annot=True, ax=ax)
        st.pyplot(fig)
        
        # Portfolio metrics
        st.subheader('Portfolio Metrics')
        metrics = calculate_metrics(
            pd.DataFrame(portfolio_data),
            market_data
        )
        st.table(metrics)
        
        # Portfolio optimization
        if st.button('Optimize Portfolio'):
            n_scenarios = st.slider('Number of scenarios', 500, 5000, 1000)
            optimize_portfolio(stock_data, tickers, n_scenarios)

def optimize_portfolio(stock_data, tickers, n_scenarios):
    """Run Markowitz portfolio optimization"""
    returns = stock_data.pct_change()
    
    results = {
        'weights': [],
        'returns': [],
        'risks': [],
        'sharpe': []
    }
    
    for _ in range(n_scenarios):
        weights = np.random.random(len(tickers))
        weights /= weights.sum()
        
        portfolio_return = (returns.mean() * weights).sum() * 252
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk
        
        results['weights'].append(weights)
        results['returns'].append(portfolio_return)
        results['risks'].append(portfolio_risk)
        results['sharpe'].append(sharpe_ratio)
    
    # Find optimal portfolio
    optimal_idx = np.argmax(results['sharpe'])
    optimal_weights = results['weights'][optimal_idx]
    
    # Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results['risks'], results['returns'], 
                        c=results['sharpe'], cmap='plasma')
    plt.colorbar(scatter, label='Sharpe Ratio')
    ax.set_xlabel('Risk (Volatility)')
    ax.set_ylabel('Expected Return')
    ax.scatter(results['risks'][optimal_idx], results['returns'][optimal_idx], 
               color='red', marker='*', s=500, label='Optimal Portfolio')
    ax.legend()
    st.pyplot(fig)
    
    # Display optimal allocation
    st.success('Optimal Portfolio Allocation:')
    optimal_allocation = pd.DataFrame({
        'Stock': tickers,
        'Weight': optimal_weights
    })
    st.table(optimal_allocation.round(4))

if __name__ == "__main__":
    main()
