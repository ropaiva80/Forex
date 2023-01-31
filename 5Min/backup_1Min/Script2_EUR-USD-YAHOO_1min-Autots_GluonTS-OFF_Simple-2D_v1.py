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
sns.set()
plt.style.use('seaborn-whitegrid')
from datetime import datetime
from datetime import timedelta
import warnings
import pyodbc
import pytz

#Datetime settings

today = datetime.today()
today += timedelta(days=2)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')

#Download of dataset
print ("Caution: Yahoo timezone set for UTC Time -7:00")
eth_df = yf.download('EURUSD=X',start_date, today, interval='1M')
eth_df.reset_index(inplace=True)
eth_df.rename(columns={'index': 'Datetime'}, inplace=True)

eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize(None)
eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize(None)

eth_df.dropna(inplace=True)
warnings.filterwarnings('ignore')
##pd.options.display.float_format = '${:,.2f}'.format


from autots import AutoTS
model = AutoTS(forecast_length=5, frequency='infer', max_generations=4, num_validations=2, model_list="superfast", ensemble=None)
model = model.fit(eth_df, date_col='Datetime', value_col='Close', id_col=None)


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