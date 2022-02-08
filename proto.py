import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

st.session_state.ticker = []
st.session_state.weight = []

ticker = st.session_state.ticker
weight = st.session_state.weight


st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

#defining some functions

@st.cache(allow_output_mutation=True)
def run(ticker,weight,data):
    ret = data.pct_change()
    x = (ret*weight).sum(axis=1)
    cum = (1+x).cumprod()
    cum.rename('Portfolio',inplace=True)
    return cum

@st.cache(allow_output_mutation=True)
def comp(data):
    df1 = yf.download('^nsei',period=f'{per}y')['Adj Close']
    ret = df1.pct_change()
    cumm = (1+ret).cumprod()
    cumm.rename('^NSEI',inplace=True)
    return cumm

st.title('Portfolio Constructor and Optimizer')
st.subheader('type ".ns" after every stock symbol')
st.text('to find stock symbols click on below link')
st.markdown('https://en.wikipedia.org/wiki/NIFTY_50#Constituents')

n = st.number_input('type the number of stocks in your portfolio',min_value=2,
                   max_value=10)
per = st.slider('period',min_value=1,max_value=10)
n = int(n)

for i in range(0,n):
    name = st.text_input(f'type the name of the stock:- {i}')
    w = st.number_input(f'type the weight of {i}')
    ticker.append(name)
    weight.append(w)
if sum(weight) != 1:
    st.warning('Warning: weight should add up to 1')
    st.text(sum(weight))
    
btn = st.button('Run')

if btn:
    st.session_state.data = yf.download(ticker,period=f'{per}y')['Adj Close']
    data = st.session_state.data
    st.session_state.portfolio = run(ticker,weight,data)
    plt.plot(st.session_state.portfolio)
    fig = plt.show()
    st.pyplot(fig)
compare = st.radio('Compare with market',['No','Yes'])

if compare == 'Yes':
    amt = st.number_input('enter initial investment')
    portfolio = st.session_state.portfolio
    data = st.session_state.data
    st.subheader(f'Your portfolio retuns vs nifty')
    comparision = comp(data)
    st.session_state.full = pd.concat([comparision,portfolio],axis=1)
    plt.plot(st.session_state.full)
    plt.legend(st.session_state.full)
    fig = plt.show()
    st.pyplot(fig)
st.header('Portfoli optimization using Markowitz model')
scenario = st.slider('No. of scenarios',min_value=100,max_value=1000)
start = st.button('Start')
if start:
    x = data.pct_change()
    p_weights = []
    p_returns = []
    p_risk = []
    p_sharpe = []
    count = scenario
    for k in range(0, count):
        wts = np.random.uniform(size = len(x.columns))
        wts = wts/np.sum(wts)
        p_weights.append(wts)

    #returns
        mean_ret = (x.mean() * wts).sum()*252
        p_returns.append(mean_ret)

    #volatility
        ret = (x * wts).sum(axis = 1)
        annual_std = np.std(ret) * np.sqrt(252)
        p_risk.append(annual_std)
    
    #Sharpe ratio
        sharpe = (np.mean(ret) / np.std(ret))* np.sqrt(252)
        p_sharpe.append(sharpe)
    max_ind = np.argmax(p_sharpe)
    max_r = np.argmax(p_returns)
    plt.scatter(p_risk, p_returns, c=p_sharpe, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')

    plt.scatter(p_risk[max_ind], p_returns[max_ind], color='r',  marker='*', s=500)
    opt = p_weights[max_ind]
    opt2 = np.round(opt,2)
    st.info(f'your portfolio allocation{ticker}-{weight}')
    st.success(f'optimum allocation {ticker} - {opt2} ')

output = plt.show()
st.pyplot(output)
