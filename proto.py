import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

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
def calculate_individual_returns(stock_data, years):
    """Calculate annualized returns for individual stocks"""
    returns = pd.DataFrame()
    
    for column in stock_data.columns:
        # Calculate total return
        total_return = ((stock_data[column].iloc[-1] / stock_data[column].iloc[0]) - 1) * 100
        # Calculate annualized return
        annualized_return = (((1 + total_return/100)**(1/years)) - 1) * 100
        # Calculate volatility
        volatility = stock_data[column].pct_change().std() * np.sqrt(252) * 100
        
        returns = returns.append({
            'Stock': column,
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2)
        }, ignore_index=True)
    
    return returns.sort_values('Annualized Return (%)', ascending=False)

# Add this inside the if analyze_button block, after the correlation heatmap:

if analyze_button:
    # [Previous analysis code remains the same...]
    
    # Individual stock returns analysis
    st.subheader('Individual Stock Performance')
    
    # Calculate and display individual stock returns
    individual_returns = calculate_individual_returns(stock_data, years)
    st.table(individual_returns)
    
    # Create bar plot of annualized returns
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(individual_returns['Stock'], individual_returns['Annualized Return (%)'])
    ax.set_title('Annualized Returns by Stock')
    ax.set_xlabel('Stocks')
    ax.set_ylabel('Annualized Return (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create scatter plot of return vs volatility
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(individual_returns['Volatility (%)'], 
              individual_returns['Annualized Return (%)'])
    
    # Add labels for each point
    for i, stock in enumerate(individual_returns['Stock']):
        ax.annotate(stock, 
                   (individual_returns['Volatility (%)'].iloc[i], 
                    individual_returns['Annualized Return (%)'].iloc[i]))
    
    ax.set_title('Risk-Return Profile of Individual Stocks')
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Cumulative returns of individual stocks
    st.subheader('Cumulative Returns Over Time')
    cum_returns = (1 + stock_data.pct_change()).cumprod()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in cum_returns.columns:
        ax.plot(cum_returns.index, cum_returns[column], label=column)
    
    ax.set_title('Cumulative Returns of Individual Stocks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def calculate_metrics(portfolio_data, market_data, years, risk_free_rate=0.0666):
    """Calculate portfolio metrics including CAPM, beta, alpha, etc."""
    # Ensure portfolio_data is a Series named 'Portfolio'
    if isinstance(portfolio_data, pd.DataFrame):
        portfolio_data = portfolio_data['Portfolio']
    
    # Calculate returns
    portfolio_returns = portfolio_data.pct_change()
    market_returns = market_data.pct_change()
    
    # Calculate beta
    covariance = portfolio_returns.cov(market_returns)
    market_variance = market_returns.var()
    beta = covariance / market_variance
    
    # Calculate CAPM expected return
    market_cagr = (((market_data.iloc[-1]/market_data.iloc[0])**(1/years))-1)*100
    capm_return = risk_free_rate + beta * (market_cagr - risk_free_rate)
    
    # Calculate actual portfolio performance
    portfolio_cagr = (((portfolio_data.iloc[-1]/portfolio_data.iloc[0])**(1/years))-1)*100
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    correlation = portfolio_returns.corr(market_returns)
    
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
            try:
                stock_data = yf.download(tickers, period=f'{years}y')['Adj Close']
                market_data = yf.download('^NSEI', period=f'{years}y')['Adj Close']
                
                if stock_data.empty or market_data.empty:
                    st.error('Failed to download data!')
                    return
            except Exception as e:
                st.error(f'Error downloading data: {str(e)}')
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
        metrics = calculate_metrics(portfolio_data, market_data, years)
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
