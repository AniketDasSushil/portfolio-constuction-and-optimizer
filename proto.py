import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# configuring the display
#defining some functions
@st.cache(allow_output_mutation=True)
def portfolio_create(ticker,weight,data):
    ret = data.pct_change()
    x = (ret*weight).sum(axis=1)
    cum = (1+x).cumprod()
    cum.rename('Portfolio',inplace=True)
    return cum
@st.cache(allow_output_mutation=True)
def cumm(data):
    ret = data.pct_change()
    cumm = (1+ret).cumprod()
    return cumm
@st.cache(allow_output_mutation=True)
def metrics(df):
    ticker = df.columns
    df1 = pd.DataFrame(df)
    df1 = df1.replace(np.nan,1)
    m_returns = (((st.session_state.market[-1]/st.session_state.market[0])**(1/yr))-1)*100 
    rf = 6.66
    ret = df1.pct_change()
    mean_ret = ret.sum()*252
    list_beta = []
    list_corr = []
    cov = ret.cov()
    corr =df1.corr()
    var = ret['Nifty'].var()
    beta = cov.loc['Portfolio','Nifty']/var
    list_beta.append(['Portfolio',beta])
    i = corr.loc['Portfolio','Nifty']
    list_corr.append(i)
    beta = pd.DataFrame(list_beta)
    beta.rename(columns = {0:'metrics',1:'beta'},inplace=True)
    list_a = []
    beta['capm'] = rf+beta['beta']*(m_returns-rf)
    capm = round(beta,2)
    list_cagr = []
    list_vol = []
    list_mean = []
    list_r = []
    cagr = (((st.session_state.portfolio[-1]/st.session_state.portfolio[0])**(1/yr))-1)*100 
    mean = ret['Portfolio'].mean()*len(ret)
    vol = round(ret['Portfolio'].std()*np.sqrt(len(ret)),2)
    list_cagr.append(cagr)
    list_vol.append(vol)
    list_mean.append(mean)
    mean_ret = ret.mean()*252
    annual_std = np.std(ret) * np.sqrt(252)
    shp = mean_ret/annual_std
    capm['cagr'] = list_cagr
    capm['volatality'] = list_vol
    capm['alpha'] = capm['cagr']-capm['capm']
    capm['mean returns'] = list_mean
    capm['correlation'] = list_corr
    capm = np.round(capm,4)
    capm = capm.astype(str)
    capm = capm.transpose()
    capm.reset_index(inplace=True)
    capm.rename(columns=capm.iloc[0],inplace=True)
    capm.drop(capm.index[0],inplace=True)
    return capm
#defining some globale variables
st.session_state.ticker = []
ticker = st.session_state.ticker
st.session_state.weight = []
st.session_state.sym = pd.read_csv('sym.csv')
sym = st.session_state.sym
sym.columns = [column.replace(" ",'_') for column in sym.columns]

st.title('Portfolio Constructor, Analyzer and Optimizer')
n = st.slider('select the number of stocks in your portfolio',min_value=2,
                   max_value=20)
yr = st.slider('number of years',min_value=1,max_value=20)
sector = []
for i in range(1,n+1):
    name = st.selectbox(f'select the stock: - {1}',sym['NAME_OF_COMPANY'],key=i)
    w = st.number_input(f'type the weight of {name}',key=i,step=5)
    tick = sym.query('NAME_OF_COMPANY == @name')
    key = tick["SYMBOL"].to_list()
    ticker.append(key[0]+".NS")
    st.session_state.weight.append(w)
    sector.append(tick['SECTOR'].to_list()[0])
    
if sum(st.session_state.weight) != 100:
    st.warning('Warning: weight should add up to 100')
    st.text(sum(st.session_state.weight))
    
btn = st.button('Confirm')


if btn:
    st.session_state.weight = st.session_state.weight/np.sum(st.session_state.weight)
    st.session_state.data = yf.download(ticker,period=f'{yr}y')['Adj Close']
    st.session_state.portfolio = portfolio_create(ticker,st.session_state.weight,st.session_state.data)
    st.session_state.market = yf.download('^NSEI',period=f'{yr}y')['Adj Close']
    plt.plot(st.session_state.portfolio)
    plt.title('cummulative returns of your portfolio')
    plt.xticks(rotation=45)
    fig = plt.show()
    st.pyplot(fig) 
compare = st.radio('compare with market',['no','yes'])
if compare == 'yes':
    nifty = cumm(st.session_state.market)
    nifty.rename('Nifty',inplace=True)
    compare = pd.concat([nifty,st.session_state.portfolio],axis=1)
    plt.plot(compare)
    plt.xticks(rotation=45)
    plt.legend(compare)
    fig = plt.show()
    st.pyplot(fig)
st.header('Portfolio Metrics')
click = st.button('Find')


if click:
   # for i in ticker:
     #   k = yf.Ticker(i)
     #   k.
    plt.pie(st.session_state.weight,labels=sector,autopct='%1.1f%%')
    plt.title('portfolio composition')
    st.pyplot(plt.show())
    st.subheader('performance of individual stocks in your portfolio')
    plt.plot(cumm(st.session_state.data))
    plt.xticks(rotation=45)
    plt.legend(cumm(st.session_state.data))
    st.pyplot(plt.show())
    st.subheader('Correlation between stocks in portfolio')
    sns.heatmap(st.session_state.data.corr(),annot=True)
    st.pyplot(plt.show())
    m = metrics(compare)
    st.table(m)
st.header('Portfolio optimization using Markowitz model')
scenario = st.slider('No. of scenarios',min_value=500,max_value=5000)
start = st.button('Start')


if start:
    x = st.session_state.data.pct_change()
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
        sharpe = ((mean_ret) / annual_std)
        p_sharpe.append(sharpe)
    max_ind = np.argmax(p_sharpe)
    plt.scatter(p_risk, p_returns, c=p_sharpe, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.scatter(p_risk[max_ind], p_returns[max_ind], color='r',  marker='*', s=500)
    opt = p_weights[max_ind]
    opt2 = np.round(opt,2)
    st.info(f'your portfolio allocation{ticker}-{st.session_state.weight}')
    st.success(f'optimum allocation {ticker} - {opt2} ')

output = plt.show()
st.pyplot(output)
