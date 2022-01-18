import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
    
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization

import seaborn as sn
import plotly.graph_objects as go
import streamlit as st
    
#Calculating RSI
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(d, com=period-1, adjust=False).mean() 
    return 100 - 100 / (1 + rs)


#Calculating EMA
def calculate_ema(prices, days, smoothing=2):
    ema = [sum(prices[:days]) / days]
    for price in prices[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema



# Calculating CCI
def calculate_cci(df,period):
    TP = df[['High','Low','Close']].mean(1)
    CCI = (TP-TP.rolling(period).mean())#/(0.015*TP.rolling(period).std())  #check the formula
    return CCI.fillna(0)


## ADX
def ADX(df, period):
    """
    Computes the ADX indicator.
    """

    # df = data.copy()
    alpha = 1 / period

    # TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # ATR
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH'] > 0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L'] > 0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM'] / df['ATR']) * 100
    df['-DMI'] = (df['S-DM'] / df['ATR']) * 100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI']) / (df['+DMI'] + df['-DMI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']

    return df['ADX']


## WRP
def get_wr(high, low, close, lookback):
    highh = high.rolling(lookback).max()
    lowl = low.rolling(lookback).min()
    wr = -100 * ((highh - close) / (highh - lowl))
    return wr


## ROC
def ROC(df, period):
    M = df.diff(period - 1)
    N = df.shift(period - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(period))
    return ROC


## SMI
def calc_SMI(df):
    high14 = df['High'].rolling(14).max()
    low14 = df['Low'].rolling(14).min()
    df['SMI'] = (df['Close'] - low14) * 100 / (high14 - low14).rolling(3).mean()
    return df


## MACD
def MACD(df, period):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


## Calculate indicators
def prepare_dataset(file_name, future_days=7, period=10):
    df = pd.read_csv(file_name, parse_dates=['Date'])
    df['Prediction'] = df['Close'].shift(-future_days)

    ## RSI
    df['RSI'] = RSI(df['Close'], period - 1)
    df.fillna(0, inplace=True)

    # EMA
    ema_vals = np.zeros(len(df))
    ema_vals[period - 1:] = calculate_ema(df['Close'], period)
    df['EMA'] = ema_vals

    ## CCI
    df['CCI'] = calculate_cci(df, period)
    df.fillna(0, inplace=True)

    # Calculating WRP
    df['WRP'] = get_wr(df['High'], df['Low'], df['Close'], period)
    df.fillna(0, inplace=True)

    # Calculating ROC
    df['ROC'] = ROC(df['Close'], period)
    df.fillna(0, inplace=True)

    ## SMI
    high14 = df['High'].rolling(14).max()
    low14 = df['Low'].rolling(14).min()
    df['SMI'] = (df['Close'] - low14) * 100 / (high14 - low14).rolling(3).mean()
    df.fillna(0, inplace=True)

    ## MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df.fillna(0, inplace=True)

    ## ADX
    df['ADX'] = ADX(df, period)
    df.fillna(0, inplace=True)

    return df


def plot_variables(df,x_var,y_vars):
    plt.figure(figsize=(18,6))
    plt.xlabel(x_var)
    plt.ylabel('Price $')
    for y_var in y_vars:
        plt.plot(df[x_var],df[y_var],label=y_var)
    plt.legend()
    plt.show()
    
    




def train_model(data,model_name,period=10,future_days=7,test_size=0.2):
    #Create the feature set and convert to numpy and remove the last x rows/days
    X = data.drop(['Date','date','Prediction'],axis=1)
    X = X.loc[period-1:len(data)-future_days-1,:]
   
    #for number in range(0,9):
    #  #  X['Close'+str(number+1)]=data['Close'].shift(-number).values[:X.shape[0]]

    #X['Average_days']=X.iloc[:,5:].mean(1)
    
    X=X.loc[:,:]
    
    #Create the target dataset 'y' and get all of the target values except for the last x rows/days
    y = data.loc[period-1:len(data)-future_days-1,'Prediction']
    y.head()
    
    #Split the data 
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=0) # change to 20
    model = model_name
    model.fit(x_train,y_train)
    y_pred_test=model.predict(x_test)

    print('Mean Squared error for',model_name,mean_squared_error(y_test, y_pred_test))
    print('Mean Absolute error for',model_name,mean_absolute_error(y_test, y_pred_test))
    
    return model,y_test,y_pred_test
    
    
    
def plot_results(y_test,y_pred):


    fig = go.Figure()

    fig.add_trace(go.Line(x=np.arange(len(y_test)),
                                 y=y_test,name='True Values'))
    fig.add_trace(go.Line(x=np.arange(len(y_test)),
                                 y=y_pred,name='Predicted Values'))

    fig.update_layout(title='Predictions',
                      title_xanchor='auto',
                      yaxis_title='Close Price',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)
    
    
    

def train_nnmodel(data,period=10,future_days=7,test_size=0.2):
    #Create the feature set and convert to numpy and remove the last x rows/days
    X = data.drop(['Date','date','Prediction'],axis=1)
    X = X.loc[period-1:len(data)-future_days-1,:]
    #for number in range(0,9):
    #    X['Close'+str(number+1)]=data['Close'].shift(-number).values[:X.shape[0]]

    #X['Average_days']=X.iloc[:,5:].mean(1)
    #X=X.loc[:,['RSI','EMA','ROC','Close']]
    
    #Create the target dataset 'y' and get all of the target values except for the last x rows/days
    y = data.loc[period-1:len(data)-future_days-1,'Prediction']
    
    #Split the data 
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=0) # change to 20
    
    model = Sequential()
    # add parameters to the model 
    model.add(Dense(input_dim = x_train.shape[1], units = 128, activation='relu'))
    model.add(Dense(units = 64, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(units = 32, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(units = 16, activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(units = 8, activation='relu'))
    model.add(Dense(units = 1))
    model.compile(loss='mse' ,optimizer='adam', metrics=['mae'])
    
    
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50)
    errors=model.evaluate(x_test,y_test)
    print('Mean Squared error for NN: ',errors[0])
    print('Mean Absolute error for NN: ',errors[1])
    y_pred_NN=model.predict(x_test)
    y_pred_trainNN=model.predict(x_train)
    
    return model, y_test,y_pred_NN


def get_covid_cases(file_name):
    covid=pd.read_csv(file_name,parse_dates=['date'])
    red_covid= covid.sort_values(by='date')[covid['date']<'2021-01-01'].groupby(by='date').sum()[['cases']].reset_index()
    return red_covid

def remove_vars(data,threshold):
    corrMatrix=data.corr()
#     display(corrMatrix)
    plt.figure(figsize = (20,5))
    matrix = sn.heatmap(corrMatrix, annot=True, linewidths=.5)
    plt.show()
    output=np.abs(corrMatrix.loc[:,'Prediction'])<threshold  
    to_drop=output[output==True].index
    print(to_drop)
    X_reduced=data.drop(to_drop,axis=1).copy()
    return X_reduced


def plot_candlesticks(df):

    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color= 'cyan', decreasing_line_color= 'gray')])

    fig.update_layout(title='Chegg',
                      title_xanchor='auto',
                     yaxis_title='Close',
                     xaxis_rangeslider_visible=True)

    fig.show()

# plot indicators
def plot_indicator(df,indicator,divide_by=1):
    fig = go.Figure()

    fig.add_trace(go.Line(x=df['Date'],
                                 y=df[indicator]/divide_by,name=indicator))
    fig.add_trace(go.Line(x=df['Date'],
                                 y=df['Close'],name='Close'))

    fig.update_layout(title=indicator,
                      title_xanchor='auto',
                      yaxis_title='Close Price($)',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)

def plot_multi_indicators(df,indicators_list,divide_by=1):
    fig = go.Figure()
    fig.add_trace(go.Line(x=df['Date'],
                                 y=df['Close'],name='Close'))
    for indicator in indicators_list:
        fig.add_trace(go.Line(x=df['Date'],
                                     y=df[indicator]/divide_by,name=indicator))

    fig.update_layout(title='Indicators Plot',
                      title_xanchor='auto',
                      yaxis_title='Close Price($)',
                      xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)