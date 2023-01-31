

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import pytz

# Some functions to help out with
def plot_predictions(test,predicted):
    plt.plot(test, color='red',label='Real IBM Stock Price')
    plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
	

#Datetime settings

today = datetime.today()
today += timedelta(days=4)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')

#Download of dataset
dataset = yf.download('AUDCHF=X',start_date, today, interval='1M')
dataset.reset_index(inplace=True)
dataset.rename(columns={'index': 'Date'}, inplace=True)


dataset['Datetime'] = dataset['Datetime'].dt.tz_localize(None)
dataset['Datetime'] = dataset['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
dataset['Datetime'] = dataset['Datetime'].dt.tz_localize(None)

##https://sparkbyexamples.com/pandas/print-pandas-dataframe-without-index/
blankIndex=[''] * len(dataset)
dataset.index=blankIndex















dataset = pdr.get_data_yahoo([IBM], index_col='Date', parse_dates=['Date'])

dataset = pdr.get_data_yahoo([IBM], 
                 start=datetime(2000, 1, 1), 
                 end=datetime.today().strftime('%Y-%m-%d'))['Close']