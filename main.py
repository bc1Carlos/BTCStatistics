import json
import datetime
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Current Code
with open("hash-rate.json", "r") as file:
    data = json.load(file)

hashrates = data["hash-rate"]
prices = data["market-price"]

def group_by_time_period(entries):
    grouped_data = {}
    for entry in entries:
        dt = datetime.datetime.utcfromtimestamp(entry["x"] / 1000)
        time_key = f"{dt.year}-{dt.month:02d}"
        grouped_data.setdefault(time_key, []).append(entry["y"])
    return grouped_data

def get_monthly_closing(grouped_data):
    # Returns the last entry for each month
    return {time_key: rates[-1] for time_key, rates in grouped_data.items()}

monthly_hashrates = group_by_time_period(hashrates)
monthly_prices = group_by_time_period(prices)
monthly_close_hashrate = get_monthly_closing(monthly_hashrates)
monthly_close_price = get_monthly_closing(monthly_prices)

df = pd.DataFrame({
    "Time": list(monthly_close_hashrate.keys()),
    "Hashrate": list(monthly_close_hashrate.values()),
    "Price": list(monthly_close_price.values())
})

# Regression of Hashrate on Price
hashrate_model = sm.OLS(df["Hashrate"], sm.add_constant(df["Price"])).fit()
print('hashrate model')
print(hashrate_model.summary())

# Regression of Price on Hashrate
price_model = sm.OLS((df["Price"]), sm.add_constant(df["Hashrate"])).fit()
print('price model')
print(price_model.summary())

df["Log_Price"] = np.log(df["Price"] + 0.0001)
# Regression of Log of Hashrate on Log of Price
hashrate_model = sm.OLS(np.log(df["Hashrate"]), sm.add_constant((df["Log_Price"]))).fit()
print('Log hashrate model')
print(hashrate_model.summary())

# Regression of Log of Price on Log of Hashrate
price_model = sm.OLS(df["Log_Price"], sm.add_constant(df["Hashrate"])).fit()
print('Log price model')
print(price_model.summary())

result = adfuller(np.log(df["Hashrate"]))
print('Augmented Dickey Fuller for Log_Hashrate')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

if result[0] < result[4]["5%"]:
    print("The Log Hashrate time series is stationary.")
else:
    print("The Log Hashrate time series is not stationary.")
    df['Hashrate_diff'] = df['Hashrate'].diff()
    df.dropna(inplace=True)  # Drop NaN values

    print("\nTesting first difference:")
    result_diff = adfuller(df['Hashrate_diff'])
    print('ADF Statistic for differenced series:', result_diff[0])
    print('p-value:', result_diff[1])
    print('Critical Values:', result_diff[4])

    if result_diff[0] < result_diff[4]["5%"]:
        print("The Log Hashrate first difference time series is stationary.")
    else:
        print("The Log Hashrate first difference time series is still not stationary.")

df["Log_Price"] = np.log(df["Price"] + 0.001)
result = adfuller((df["Log_Price"]))
print('Augmented Dickey Fuller for Log_Price')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

if result[0] < result[4]["5%"]:
    print("The Log Price time series is stationary.")
else:
    print("The Log Price time series is not stationary.")
    df['Log_Price_diff'] = df['Log_Price'].diff()
    df.dropna(inplace=True)  # Drop NaN values

    print("\nTesting first difference:")
    result_diff = adfuller(df['Log_Price_diff'])
    print('Augmented Dickey Fuller for Log_Price first difference')
    print('ADF Statistic for differenced series:', result_diff[0])
    print('p-value:', result_diff[1])
    print('Critical Values:', result_diff[4])

    if result_diff[0] < result_diff[4]["5%"]:
        print("The Log Price first difference time series is stationary.")
    else:
        print("The Log Price first difference time series is still not stationary.")

# Autoregressive Model (AR) for Hashrate Prediction
ar_model = AutoReg(df["Hashrate"], lags=2).fit()
print('hashrate AR model')
print(ar_model.summary())

ar_model_log = AutoReg(np.log(df["Hashrate"]), lags=2).fit()
print('log of hashrate AR model')
print(ar_model_log.summary())

month_predict = 60
future_hashrate_ar = ar_model.predict(start=len(df), end=len(df) + month_predict)

ar_model_log = AutoReg(np.log(df["Hashrate"]), lags=1).fit()
future_hashratelog_ar = ar_model_log.predict(start=len(df), end=len(df) + month_predict)

df["Log_Price"] = np.log(df["Price"] + 0.001)
# 2. Develop SARIMAX model using log-transformed values
sarimax_log_price_model = sm.tsa.SARIMAX((df["Log_Price"]),
                                         exog=np.log(df["Hashrate"]),
                                         order=(1, 1, 1),
                                         seasonal_order=(1, 1, 0, 48),  # 48 months for the 4-year halving cycle
                                         enforce_stationarity=False,
                                         enforce_invertibility=False).fit()

print('SARIMAX Log Model')
print(sarimax_log_price_model.summary())
# 3. Predict future prices using the log-transformed hashrate predictions
predicted_log_prices = sarimax_log_price_model.predict(start=len(df), end=len(df) + month_predict, exog=future_hashratelog_ar.values.reshape(-1,1))

# Predict Future Prices using AR-predicted Hashrate
future_dates_count = len(predicted_log_prices)
future_dates = pd.date_range(start="2023-09", periods=future_dates_count, freq="MS")

# 4. Store the log-transformed predictions in a new dataframe and save to CSV
df_log_future = pd.DataFrame({
    "Time": future_dates,
    "Log_Predicted_Hashrate": future_hashratelog_ar.values,
    "Log_Predicted_Price": predicted_log_prices.values
})

df_log_future.to_csv('prediction-model.csv', index=False)

# Drop NaN values, just in case
df.dropna(inplace=True)

# Granger causality test for "Price" on "Hashrate" (does Price Granger cause Hashrate?)
print("Testing if Price Granger causes Hashrate:")
#gc_price_on_hashrate = grangercausalitytests(df[['Hashrate', 'Price']], maxlag=12, verbose=True)

# Granger causality test for "Hashrate" on "Price" (does Hashrate Granger cause Price?)
print("Testing if Hashrate Granger causes Price:")
#gc_hashrate_on_price = grangercausalitytests(df[['Price', 'Hashrate']], maxlag=12, verbose=True)

# Assuming you've computed Log_Hashrate and stored in df
df['Log_Hashrate'] = np.log(df['Hashrate'])

# Plot ACF
sm.graphics.tsa.plot_acf(df['Log_Hashrate'], lags=24)
plt.title("Autocorrelation Function for Log_Hashrate")
plt.show()

# Plot ACF
df["Log_Price"] = np.log(df["Price"] + 0.001)
sm.graphics.tsa.plot_acf(df['Log_Price'], lags=24)
plt.title("Autocorrelation Function for Log_Price")
plt.show()

sm.graphics.tsa.plot_pacf(df['Log_Hashrate'], lags=24)
plt.title("Partial Autocorrelation Function for Log_Hashrate")
plt.show()
