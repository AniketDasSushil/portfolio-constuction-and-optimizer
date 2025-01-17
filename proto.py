import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Previous functions remain the same...
[Previous code for portfolio_create, calculate_cumulative_returns, and calculate_metrics functions stays exactly the same]

def optimize_portfolio(stock_data, tickers, n_scenarios):
    """Run Markowitz portfolio optimization"""
    returns = stock_data.pct_change()
    
    results = {
        'weights': [],
        'returns': [],
        'risks': [],
        'sharpe': []
    }
    
    with st.spinner('Running optimization...'):
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
            'Weight': optimal_weights * 100  # Convert to percentage
        })
        st.table(optimal_allocation.round(2))

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
    
    # Create two columns for the buttons
    col1, col2 = st.columns(2)
    analyze_button = col1.button('Analyze Portfolio')
    optimize_button = col2.button('Optimize Portfolio')
    
    if analyze_button or optimize_button:
        weights = np.array(weights) / 100  # Normalize weights
        
        # Download data
        with st.spinner('Downloading data...'):
            try:
                stock_data = yf.download(tickers, period=f'{years}y')['Adj Close']
                market_data = yf.download('^NSEI', period=f'{years}y')['Adj Close']
                
                if stock_data.empty or market_data.empty:
                    st.error('Failed to download data!')
                    return
            except Exception as e:
                st.error(f'Error downloading data: {str(e)}')
                return
        
        if analyze_button:
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
            metrics = calculate_metrics(portfolio_data, market_data, years)
            st.table(metrics)
        
        if optimize_button:
            st.subheader('Portfolio Optimization')
            n_scenarios = st.slider('Number of scenarios', 500, 5000, 1000)
            optimize_portfolio(stock_data, tickers, n_scenarios)

if __name__ == "__main__":
    main()
