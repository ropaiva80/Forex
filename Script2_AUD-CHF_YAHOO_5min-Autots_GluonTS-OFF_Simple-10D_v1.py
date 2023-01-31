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
from sqlalchemy import create_engine
import pymysql


#Datetime settings

today = datetime.today()
today += timedelta(days=4)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=10)
start_date = start_date.strftime('%Y-%m-%d')

#Download of dataset
print ("Caution: Yahoo timezone set for UTC Time -7:00")
eth_df = yf.download('AUDCHF=X',start_date, today, interval='5M')
eth_df.reset_index(inplace=True)
eth_df.rename(columns={'index': 'Datetime'}, inplace=True)

eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize(None)
eth_df['Datetime'] = eth_df['Datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
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

##Change the labels of columns and drop exceeded column ID
df.columns = ['AUD_DATETIME_PRED', 'AUD_PRICE_PRED', 'ID', 'AUD_DIF']
df.drop(columns='ID', inplace=True)

##Add columns and constant values
df['AUD_NAM'] = 'AUD_CHF'
df['AUD_ALG'] = 'Script2_AutoTS'
df['AUD_CAT'] = 'Forex'
print(df)


##DATABASE##
## Record directly from DATAFRAME to DATABASE
my_conn=create_engine('mysql+mysqldb://forex:F0rex2022@usarizona_web:30307/Forex')
df.to_sql(con=my_conn,name='AUD_CHF',if_exists='append',index=False)

clock = datetime.now()
print (clock.strftime("%Y-%m-%d %H:%M:%S"))

#for index, row in df.iterrows():
#    mysql_insert_query("INSERT INTO AUD_CHF (AUD_DATETIME_PRED,AUD_PRICE_PRED,AUD_DIF,AUD_NAM,AUD_ALG,AUD_CAT) values(?,?,?,'FOREX-AUDCHF','SCRIPT2_AutoTS','5min')", row.time, row.Close, row.difference)
#    record = (AUD_DATETIME_PRED,AUD_PRICE_PRED,AUD_DIF,AUD_NAM,AUD_ALG,AUD_CAT)
#    cursor.execute(mysql_insert_query, record)
#    db.connection.commit()
#    print("Record inserted successfully into Laptop table")
#cursor.close()


