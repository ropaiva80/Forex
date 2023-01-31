
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import pytz

#Datetime settings

today = datetime.today()
today += timedelta(days=4)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')

#Download of dataset
dataset = yf.download('BTC-USD',start_date, today, interval='1M')
dataset.reset_index(inplace=True)
dataset.rename(columns={'index': 'Date'}, inplace=True)


dataset['Datetime'] = dataset['Datetime'].dt.tz_localize(None)
dataset['Datetime'] = dataset['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
dataset['Datetime'] = dataset['Datetime'].dt.tz_localize(None)

## 1 Choice
##https://sparkbyexamples.com/pandas/print-pandas-dataframe-without-index/
#blankIndex=[''] * len(dataset)
#dataset.index=blankIndex

## 2 Choice
data = dataset
data.index = pd.to_datetime(data['Datetime'])
data.drop(columns='Datetime',inplace=True)
data.head()

