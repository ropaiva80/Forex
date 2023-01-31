# Python ML#
# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

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

#Timezone reset
import pytz

# yahoo finance is used to fetch data
import yfinance as yf

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Read data
# Don't forget setting up for one day ahead!!

today = datetime.today()
today += timedelta(days=2)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=6)
start_date = start_date.strftime('%Y-%m-%d')


# Collect data from Yahoo (hourly)
Df = yf.download('GBPJPY=X', start_date, today, interval='1M', auto_adjust=True)

# Only keep close columns
Df = Df[['Close']]

# Drop rows with missing values
Df = Df.dropna()


############## Caution: Very important (hourly/minutes predict) ##################
############## Reset Index (hourly/minutes standard) ##############

Df = Df.tz_localize(None)
Df = Df.tz_localize('UTC').tz_convert('US/Pacific')
Df = Df.tz_localize(None)

# Plot the closing price of GLD
Df.Close.plot(figsize=(10, 7),color='r')
##plt.ylabel("BTC Prices")
##plt.title("BTC Price Series")
##plt.show()

# Define explanatory variables
# An explanatory variable is a variable that is manipulated to determine the value of the BTC ETF price the next day.
# The explanatory variables in this strategy are the moving averages for past 3 days and 9 days.
# However, you can add more variables to X which you think are useful to predict the prices of the BTC ETF. 
# These variables can be technical indicators, the price of another ETF such as BTC miners ETF (GDX) or Oil ETF (USO), or US economic data.

Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_hour_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
# Similarly, the dependent variable depends on the values of the explanatory variables. 
y = Df['next_hour_price']

# Split the data into train and test dataset
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]


# Create a linear regression model

linear = LinearRegression().fit(X_train, y_train)
print("Linear Regression model")
print("BTC Price (y) = %.2f * 3 Days Moving Average (x1) \
+ %.2f * 9 Days Moving Average (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))

# Predicting the BTC ETF prices

predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(10, 7))
y_test.plot()
##plt.legend(['predicted_price', 'actual_price'])
##plt.ylabel("BTC Price")
##plt.show()

# R square
# Compute the goodness of the fit using the score() function.

r2_score = linear.score(X[t:], y[t:])*100
float("{0:.2f}".format(r2_score))


## Plotting cumulative returns

BTC = pd.DataFrame()

BTC['price'] = Df[t:]['Close']
BTC['predicted_price_next_hour'] = predicted_price
BTC['actual_price_next_day'] = y_test
BTC['BTC_returns'] = BTC['price'].pct_change().shift(-1)

BTC['signal'] = np.where(BTC.predicted_price_next_hour.shift(1) < BTC.predicted_price_next_hour,1,0)

BTC['strategy_returns'] = BTC.signal * BTC['BTC_returns']
((BTC['strategy_returns']+1).cumprod()).plot(figsize=(10,7),color='g')
##plt.ylabel('Cumulative Returns')
##plt.show()


# Calculate sharpe ratio
sharpe = BTC['strategy_returns'].mean()/BTC['strategy_returns'].std()*(252**0.5)
'Sharpe Ratio %.2f' % (sharpe)


## Predict step
# Get the data

today = datetime.today()
today += timedelta(days=2)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=6)
start_date = start_date.strftime('%Y-%m-%d')


# Collect data from Yahoo (hourly)
data = yf.download('GBPJPY=X', start_date, today, interval='1M', auto_adjust=True)

# Setting Timezone from UTC to US/Pacific
data = data.tz_localize(None)
data = data.tz_localize('UTC').tz_convert('US/Pacific')
data = data.tz_localize(None)


data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()

# Forecast the price
data['predicted_BTC_price'] = linear.predict(data[['S_3', 'S_9']])
data['signal'] = np.where(data.predicted_BTC_price.shift(1) < data.predicted_BTC_price,"Buy","No Position")

# Print the forecast
data.tail(4)[['signal','predicted_BTC_price']].T

print ("CAUTION: Timezone -07:00 - UTC - Prediction 1Min ahead")
clock = datetime.now()
print (clock.strftime("%Y-%m-%d %H:%M:%S"))


df = data.tail(4)
df = df.reset_index()
df['ID'] = df.index

df.rename(columns={'predicted_BTC_price': 'Price_prediction'}, inplace=True)
df.rename(columns={'index': 'Time'}, inplace=True)
df.rename(columns={'signal': 'Position'}, inplace=True)

df['Difference'] = df['Price_prediction'].diff()
df['Difference'] = df['Difference'].fillna(0)
print (df)
