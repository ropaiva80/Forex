import pandas as pd
import numpy as np
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
import tensorflow as tf

# Data preparation
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# AI
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# SETTINGS

# Window size or the sequence length, 7 (1 week)
N_STEPS = 5

# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]

# Stock ticker
STOCK = 'EURCAD=X'

# Current date
today = datetime.today()
today += timedelta(days=2)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=6)
start_date = start_date.strftime('%Y-%m-%d')

# Collect data from Yahoo (hourly)
init_df = yf.download(STOCK, start_date, today, interval='1M', auto_adjust=True)	

############## Reset Index (hourly/minutes standard) ##############

init_df = init_df.tz_localize(None)
init_df = init_df.tz_localize('UTC').tz_convert('US/Pacific')
init_df = init_df.tz_localize(None)

# remove columns which our neural network will not use
init_df = init_df.drop(['Open', 'High', 'Low','Volume'], axis=1)
# create the column 'date' based on index column
init_df['date'] = init_df.index	

# Let's preliminary see our data on the graphic
##plt.style.use(style='ggplot')
##plt.figure(figsize=(16,10))
##plt.plot(init_df['Close'][-100:])
##plt.xlabel("hours")
##plt.ylabel("price")
##plt.legend([f'Actual price for {STOCK}'])
##plt.show()

# Scale data for ML engine
scaler = MinMaxScaler()
init_df['Close'] = scaler.fit_transform(np.expand_dims(init_df['Close'].values, axis=1))

def PrepareData(days):
 df = init_df.copy()
 df['future'] = df['Close'].shift(-days)
 last_sequence = np.array(df[['Close']].tail(days))
 df.dropna(inplace=True)
 sequence_data = []
 sequences = deque(maxlen=N_STEPS)
 for entry,target in zip(df[['Close'] + ['date']].values, df['future'].values):sequences.append(entry)
 if len(sequences) == N_STEPS:
    sequence_data.append([np.array(sequences), target])
    last_sequence = list([s[:len(['Close'])] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
# construct the X's and Y's
 X,Y = [], []
 for seq, target in sequence_data:
  X.append(seq)
  Y.append(target)
  # convert to numpy arrays
  X = np.array(X)
  Y = np.array(Y)
  return df, last_sequence, X, Y

 
def GetTrainedModel(x_train, y_train):
 model = Sequential()
 model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['Close']))))
 model.add(Dropout(0.3))
 model.add(LSTM(120, return_sequences=False))
 model.add(Dropout(0.3))
 model.add(Dense(20))
 model.add(Dense(1))
 BATCH_SIZE = 8
 EPOCHS = 80
 model.compile(loss='mean_squared_error', optimizer='adam')
 model.fit(x_train, y_train,
           batch_size=BATCH_SIZE,
           epochs=EPOCHS,
           verbose=1)
 model.summary()
 return model

# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
   df, last_sequence, x_train, y_train = PrepareData(step)
   x_train = x_train[:, :, :len(['Close'])].astype(np.float32)
   model = GetTrainedModel(x_train, y_train)
   last_sequence = last_sequence[-N_STEPS:]
   last_sequence = np.expand_dims(last_sequence, axis=0)
   prediction = model.predict(last_sequence)
   predicted_price = scaler.inverse_transform(prediction)[0][0]
   predictions.append(round(float(predicted_price), 2))

if bool(predictions) == True and len(predictions) > 0:
     predictions_list = [str(d)+'$' for d in predictions]
     predictions_str = ', '.join(predictions_list)
     message = f'{STOCK} prediction for upcoming 3 minutes ({predictions_str})'
     print(message)
     
### walking at here ###

copy_df = init_df.copy()
y_predicted = model.predict(x_train)
y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
copy_df = copy_df.tail(3)
copy_df[f'predicted_close'] = y_predicted_transformed
copy_df


# Add predicted results to the table

date_now = datetime.now()
date_tomorrow = date_now + dt.timedelta(minutes = 1)
date_after_tomorrow = date_now + dt.timedelta(minutes = 2)

date_now = date_now.strftime("%H:%M:%S")
date_tomorrow = date_tomorrow.strftime("%H:%M:%S")
date_after_tomorrow = date_after_tomorrow.strftime("%H:%M:%S")


copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0, 0]
copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0, 0]
copy_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0, 0]

# Result chart
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(copy_df['close'][-150:].head(147))
plt.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
plt.plot(copy_df['close'][-150:].tail(4))
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {STOCK}', 
            f'Predicted price for {STOCK}',
            f'Predicted price for future 3 days'])
plt.show()