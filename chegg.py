
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from helpers import *

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import plotly.express as px
import scipy
import itertools


st.title("Stock Market Prediction")


selectbox=st.selectbox("Select dataset for prediction",['Chegg','Boxl','Stride','TWOU'])
st.write(f"You selected {selectbox}")
futuredays = st.slider("Days of prediction:", 2,7)
# nperiod = n_period *10
n_period=futuredays  # This parameters is used to calculate the indicators

st.write('Selected number of days: ',futuredays)


st.subheader('Raw Data')
raw_data=prepare_dataset(selectbox.lower()+'.csv',future_days=futuredays,period=n_period)
#raw_data=raw_data.iloc[:-futuredays,:]
st.write(raw_data)



#st.write('Time Series Data')
#st.line_chart(raw_data[['Open','Close']])

df=raw_data.copy()

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     increasing_line_color='cyan', decreasing_line_color='gray')])

fig.update_layout(title=selectbox,
                  title_xanchor='auto',
                  yaxis_title='Close',
                  xaxis_rangeslider_visible=True)

st.plotly_chart(fig)


## Select indicator to plot
st.subheader('Display indicators')
indicators_plot=[]
checkbox_rsi = st.checkbox('RSI: ',help='Relative Strength Index - RSI - The relative strength index (RSI) is a technical momentum indicator that compares the magnitude of recent gains to recent losses in an attempt to determine overbought and oversold conditions of an asset. RSI is mostly used to help traders identify momentum, market conditions and warning signals for dangerous price movements. ')
checkbox_ema =st.checkbox('EMA:', help = 'Exponential Moving Average - EMA An exponential moving average (EMA) is a type of moving average that is similar to a simple moving average, except that more weight is given to the latest data. The exponential moving average is also known as "exponentially weighted moving average". ')
checkbox_cci = st.checkbox('CCI:', help='The Commodity Channel Index (CCI) is a technical indicator that measures the difference between the current price and the historical average price. When the CCI is above zero, it indicates the price is above the historic average. Conversely, when the CCI is below zero, the price is below the historic average.')
checkbox_macd = st.checkbox('MACD:',help='MACD is an indicator that detects changes in momentum by comparing two moving averages. It can help traders identify possible buy and sell opportunities around support and resistance levels.‘Convergence’ means that two moving averages are coming together, while ‘divergence’ means that they’re moving away from each other. ')
checkbox_smi = st.checkbox('SMI:', help='Stochastic Momentum Index - SMI The Stochastic oscillator is a technical momentum indicator that compares a securitys closing price to its price range over a given timeframe.')
checkbox_adx = st.checkbox('ADX:', help='The ADX illustrates the strength of a price trend. It works on a scale of 0 to 100, where a reading of more than 25 is considered a strong trend, and a number below 25 is considered a drift. Traders can use this information to gather whether an upward or downward trend is likely to continue.ADX is normally based on a moving average of the price range over 14 days, depending on the frequency that traders prefer. Note that ADX never shows how a price trend might develop, it simply indicates the strength of the trend. The average directional index can rise when a price is falling, which signals a strong downward trend')
checkbox_wrp = st.checkbox('WRP: ', help='Williams % R - WPR Williams %R, in technical analysis, is a momentum indicator measuring overbought and oversold levels, similar to a stochastic oscillator. It was developed by Larry Williams and compares a stocks close price to the high-low range')
checkbox_roc = st.checkbox('ROC: ', help = 'The price rate of change (ROC) is a technical indicator that measures the percentage change between the most recent price and the price "n" times in the past. It is calculated by using the following formula: (Closing Price Today - Closing Price "n" times Ago) / Closing Price "n" times Ago ROC is classed as a price momentum indicator or a velocity indicator because it measures the rate of change or the strength of momentum of change. ')


if checkbox_rsi:
    st.write('You selected: RSI')
    #plot_indicator(df,indicator='RSI')
    indicators_plot.append('RSI')
if checkbox_ema:
    st.write('You selected: EMA')
    #plot_indicator(df, indicator='EMA')
    indicators_plot.append('EMA')
if checkbox_cci:
    st.write('You selected: CCI')
    #plot_indicator(df, indicator='CCI')
    indicators_plot.append('CCI')
if checkbox_macd:
    st.write('You selected: MACD')
    #plot_indicator(df, indicator='MACD')
    indicators_plot.append('MACD')
if checkbox_smi:
    st.write('You selected: SMI')
    #plot_indicator(df, indicator='SMI')
    indicators_plot.append('SMI')
if checkbox_wrp:
    st.write('You selected: WRP')
    #plot_indicator(df, indicator='WRP')
    indicators_plot.append('WRP')
if checkbox_roc:
    st.write('You selected: ROC')
    #plot_indicator(df, indicator='ROC')
    indicators_plot.append('ROC')

if checkbox_adx:
    st.write('You selected: ADX')
    #plot_indicator(df, indicator='ADX')
    indicators_plot.append('ADX')

plot_multi_indicators(df, indicators_list=indicators_plot)



## Include covid cases
st.subheader('Covid Cases with respect to Close Price for 2019 only. Covid cases are based on USA records only.')
red_covid=get_covid_cases('us-counties.csv')
red_trade=df[df['Date']>'2019-12-31']
red_df = pd.merge(red_trade, red_covid, left_on='Date', right_on='date')
if selectbox=='Boxl':
    divide_by=2000000 # due to the scale issues
else:
    divide_by=200000
plot_indicator(red_df, indicator='cases',divide_by=divide_by)

case_close_corr,_=scipy.stats.pearsonr(red_df['cases'],red_df['Close'])
st.write('Correlation between Cases and Close Price: ',case_close_corr)


## Correlation matrix can allow to select indicators
st.subheader('Correlation matrix')
corrMatrix = red_df.loc[:,['ADX','RSI','EMA','CCI','ROC','cases','WRP','MACD','SMI','Prediction','Close']].corr()
fig = px.imshow(corrMatrix)
st.plotly_chart(fig)

st.subheader('Prepare for training 80/20')
## Train ML Model
# Create the feature set and convert to numpy and remove the last x rows/days
st.write('Select indicators to include for Training')

checkbox_rsi1 = st.checkbox('Indicator RSI')
checkbox_ema1 = st.checkbox('Indicator EMA')
checkbox_cci1 = st.checkbox('Indicator CCI')
checkbox_macd1 = st.checkbox('Indicator MACD')
checkbox_smi1 = st.checkbox('Indicator SMI')
checkbox_adx1 = st.checkbox('Indicator ADX')
checkbox_wrp1 = st.checkbox('Indicator WRP')
checkbox_roc1 = st.checkbox('Indicator ROC')
checkbox_cases1 = st.checkbox('Cases')

all_attributes=[]
all_attributes.extend(['Close','Open','High','Low'])
data=df.copy()

if checkbox_rsi1:
    all_attributes.append('RSI')
if checkbox_ema1:
    all_attributes.append('EMA')
if checkbox_cci1:
    all_attributes.append('CCI')
if checkbox_macd1:
    all_attributes.append('MACD')
if checkbox_smi1:
    all_attributes.append('SMI')
if checkbox_adx1:
    all_attributes.append('ADX')
if checkbox_wrp1:
    all_attributes.append('WRP')
if checkbox_roc1:
    all_attributes.append('ROC')

if checkbox_cases1:
    all_attributes.append('cases')
    data=red_df.copy()

st.write('Features Selected',all_attributes)

period=n_period
st.write('Dataset size',len(data))

X = data.loc[period - 1:len(data)-futuredays-1,all_attributes]
y=data.loc[period-1:len(data)-futuredays-1,'Prediction']
# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
 # Select model
model_selection=st.radio('Select Machine Learning Model for Training',('SVM','Random Forest','Neural Networks'))

if model_selection=='SVM':

    model = SVR()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_all = model.predict(X)

    st.write('MSE for SVM',mean_squared_error(y_test, y_pred))
    st.write('MAE for SVM',mean_absolute_error(y_test, y_pred))
    plot_results(y_test, y_pred)


elif model_selection=='Random Forest':

    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_all = model.predict(X)

    st.write('MSE for RF', mean_squared_error(y_test, y_pred))
    st.write('MAE for RF', mean_absolute_error(y_test, y_pred))
    plot_results(y_test, y_pred)

elif model_selection =='Neural Networks':

    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(units=32, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(units=16, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)
    errors = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    y_pred_all=model.predict(X).reshape(-1)

    st.write('MSE for NN', mean_squared_error(y_test, y_pred))
    st.write('MAE for NN', mean_absolute_error(y_test, y_pred))
    plot_results(y_test, y_pred.reshape(-1))

output=data.loc[period - 1:len(data)-futuredays-1,:].copy()
output['Model_Prediction']=y_pred_all.round(3)
output['Residual']=np.abs(output['Prediction']- output['Model_Prediction'])
output['Residual_percen']=1-output['Residual']/output['Prediction']

st.subheader('Results')
st.write(output[['Date','Prediction','Model_Prediction','Residual','Residual_percen']])





