import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima.model_selection import train_test_split

stock = yf.Ticker('VTS.AX')
btchistory = stock.history(start='2019-01-01', end='2024-09-10')
btchistory.to_csv('VTShistory.csv')

stock = pd.read_csv('VTShistory.csv')
stock['Date'] = pd.to_datetime(stock['Date'], utc=True)
stock.set_index(stock['Date'].dt.date, inplace=True)
del stock['Date']
print(stock.tail())

# Plot the BTC prices
# plt.plot(stock.index, stock['Close'])
# plt.ylabel('BTC Price')
# plt.xlabel('Date')
# plt.xticks(rotation=45)
# plt.show()

# Set up training data
train = stock[stock.index < pd.Timestamp('2024-09-11').date()]
test = stock[stock.index >= pd.Timestamp('2024-09-01').date()]

plt.plot(train['Close'], color = "black")
plt.plot(test['Close'], color = "red")
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for BTC Data")

# # Autoregressive moving average model
# y = train['Close']
# ARMAmodel = SARIMAX(y, order=(1, 0, 1))
# ARMAmodel = ARMAmodel.fit()

# # Train
# y_pred = ARMAmodel.get_forecast(len(test.index))
# y_pred_df = y_pred.conf_int(alpha = 0.05) 
# y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
# y_pred_df.index = test.index
# y_pred_out = y_pred_df["Predictions"] 

# plt.plot(y_pred_out, color='green', label = 'Predictions')
# plt.legend()
# plt.show()

# # Calculate R squared value
# # arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
# # print("RMSE: ",arma_rmse)

# # # ARIMA Model
# # ARIMAmodel = ARIMA(y, order = (5, 4, 2))
# # ARIMAmodel = ARIMAmodel.fit()

# # # Same stuff as before
# # y_pred = ARIMAmodel.get_forecast(len(test.index))
# # y_pred_df = y_pred.conf_int(alpha = 0.05) # Set confidence interval
# # y_pred_df["Predictions"] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
# # y_pred_df.index = test.index
# # y_pred_out = y_pred_df["Predictions"] 

# # plt.plot(y_pred_out, color='Yellow', label = 'ARIMA Predictions')
# # plt.legend()


# # # Calculate R squared value
# # arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
# # print("RMSE: ",arma_rmse)

# Seasonal ARIMA Model
y = train['Close']
SARIMAXmodel = SARIMAX(y, order = (1, 1, 1), seasonal_order=(2,2,2,12))
SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(test.index)+30)
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = y_pred.predicted_mean
y_pred_df.index = test.index.union(pd.date_range(start=test.index[-1], periods=31, freq='D'))[1:]
y_pred_out = y_pred_df["Predictions"]
plt.plot(y_pred_out, color='Blue', label = 'SARIMA Predictions')
plt.legend()
plt.show()


arma_rmse = np.sqrt(mean_squared_error(test["Close"].values, y_pred_df["Predictions"]))
print(f"RMSE: {arma_rmse}")


# # # Split the data into training and test sets
# # train, test = train_test_split(btc['Close'], train_size=0.8)

# # # Fit the model
# # model = pm.auto_arima(train, seasonal=True, m=12,
# #                       start_p=1, start_q=1, max_p=5, max_q=5,
# #                       d=None, trace=True,
# #                       error_action='ignore',  # don't want to know if an order does not work
# #                       suppress_warnings=True, 
# #                       stepwise=True)

# # # # Print the model summary
# # # print(model.summary())

# # # Make predictions
# # predictions = model.predict(n_periods=len(test))
# # plt.plot(predictions, color='green', label='Predictions')
# # plt.show()