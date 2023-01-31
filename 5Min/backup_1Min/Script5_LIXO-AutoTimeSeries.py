## Historic-Crypto 0.1.6 ##
## An open source Python library for the collection of Historical Cryptocurrency data. ##
## https://pypi.org/project/Historic-Crypto/
## Copy table SQL - https://docs.microsoft.com/en-us/sql/relational-databases/tables/duplicate-tables?view=sql-server-ver15
## C:\Users\ropai\anaconda3\envs\
## conda activate cryptov2-37

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from seaborn import regression
from auto_ts import auto_timeseries
sns.set()
plt.style.use('seaborn-whitegrid')
from datetime import datetime
from datetime import timedelta
import warnings
import pyodbc
import pytz

#Datetime settings

today = datetime.today()
today += timedelta(days=4)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')

#Download of dataset
print ("Caution: Yahoo timezone set for UTC Time -7:00")
eth_df = yf.download('AUDCHF=X',start_date, today, interval='1M')
eth_df.reset_index(inplace=True)
eth_df.rename(columns={'index': 'Datetime'}, inplace=True)

eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize(None)
eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize(None)

eth_df.dropna(inplace=True)
pd.options.display.float_format = '${:,.4f}'.format
warnings.filterwarnings('ignore')

eth_df = eth_df.drop(["Open","Low", "Adj Close","High", "Volume"], axis = 1)
eth_df['Datetime'] = pd.to_datetime(eth_df['Datetime'])
eth_df= eth_df.sort_values('Datetime')


####eth_df = eth_df.assign(Index=range(len(eth_df))).set_index('Datetime')


####working here:#################################
# Math - method for training with 80% of dataset #
n1=90 #training
n2=10 #testing

train_df    = eth_df.head(int(len(eth_df)*(n1/100)))
test_eth_df = eth_df.tail(int(len(eth_df)*(n2/100)))

## plot
train_df.Close.plot(figsize=(15,8), title= 'AUDCHF Price', fontsize=14, label='Train')
test_eth_df.Close.plot(figsize=(15,8), title= 'AUDCHF Price', fontsize=14, label='Test')
plt.legend()
plt.grid()
plt.show()

model = auto_timeseries( score_type='rmse', time_interval="1min", forecast_period=5, non_seasonal_pdq=(3,1,3), seasonality=True, seasonal_period=1,model_type=['SARIMAX'], verbose=2)
model.fit(traindata= train_df, ts_column="Datetime", target="Close")

model.get_leaderboard()
model.plot_cv_scores()

future_predictions = model.predict(5)












prediction = model.predict()
forecast = prediction.forecast
print(forecast)










df = forecast
df = df.reset_index()
df.rename(columns={'index': 'time'}, inplace=True)
df['ID'] = df.index
df['difference'] = df['Close'].diff()
df['difference'] = df['difference'].fillna(0)
print ("Caution: Yahoo timezone set for UTC Time -7:00")
print(df)


clock = datetime.now()
print (clock.strftime("%Y-%m-%d %H:%M:%S"))