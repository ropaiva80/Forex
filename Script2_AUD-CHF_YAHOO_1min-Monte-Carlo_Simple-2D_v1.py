import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
from random import random
from seaborn import regression
from datetime import datetime
from datetime import timedelta
import warnings
import pyodbc
import pytz

##ticker = yf.ticker(
today = datetime.today()
today += timedelta(days=4)
today = today.strftime('%Y-%m-%d')

init_time_now = datetime.now()
start_date = init_time_now - timedelta(days=1)
start_date = start_date.strftime('%Y-%m-%d')
ticker = yf.download('AUDCHF=X',start_date, today, interval='1M')
ticker.reset_index(inplace=True, level=0)

ticker.rename(columns={'index': 'Datetime'}, inplace=True)

ticker['Datetime'] = ticker['Datetime'].dt.tz_localize(None)
ticker['Datetime'] = ticker['Datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
ticker['Datetime'] = ticker['Datetime'].dt.tz_localize(None)

# Set Datetime column as index of dataset
hist = ticker.set_index('Datetime')
hist = hist[['Close']]

#Plotting Price Data
hist['Close'].plot(title="AUDCHF Stock Price", ylabel = "Closing Price [$]", figsize=[10,6])

plt.grid()

minutes = [i for i in range(1, len(hist['Close'])+1)]
price_orig = hist['Close'].tolist()
change = hist['Close'].pct_change().tolist()
change = change[1:]

mean = np.mean(change)
std_dev = np.std(change)
print('Mean percent change: ' + str(round(mean*100, 2)) + '%')
print('Standard Deviation of percent change: '+ str(round(std_dev*100, 2)) + '%')

#Simulation number and prediction period
simulations = 200
minutes_to_sim = 1*2

#Initializing figure for simulations
fig = plt.figure(figsize=[10, 6])
plt.plot(minutes, price_orig)
plt.title("Monte Carlo AUDCHF Prices [" + str(simulations) + " simulations]")
plt.xlabel("Trading Days After" + start_date)
plt.ylabel("Closing Price [$]")
plt.xlim([200, len(minutes)+minutes_to_sim])
plt.grid()

#Initializing list for analytics
close_end = []
above_close = []

# For loop number of simulations desired
for i in range(simulations):
    num_minutes = [minutes[-1]]
    close_price = [hist.iloc[-1,0]]
    
    #For loop for number of days to prediction
    for j in range(minutes_to_sim):
        num_minutes.append(num_minutes[-1]+1)
        perc_change = norm.ppf(random(), loc=mean, scale=std_dev)
        close_price.append(close_price[-1]*(1+perc_change))
        
        if close_price[-1] > price_orig[-1]:
            above_close.append(1)
        else:
            above_close.append(0)
        
        close_end.append(close_price[-1])
        plt.plot(num_minutes, close_price)

#Average Closing Price and Probability of Increasing After 1 year

average_closing_price = sum(close_end)/simulations
average_perc_change = (average_closing_price-price_orig[-1])/price_orig[-1]

probability_of_increase = sum(above_close)/simulations
print('Predicted closing price after ' + str(simulations) + ' simulations: $' + str(round(average_closing_price, 2)))
print('Probability of AUDCHF price increasing after next minutes: ' + str(round(probability_of_increase*100, 2)) + '%')

#Display results:

plt.show()
    
        

