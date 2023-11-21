# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:25:51 2023

@author: fajka
"""

import eikon as ek
import numpy as np
import arch
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Set your app key
ek.set_app_key('37934fabc6604c9bb80b65b1e3fa6bd265ad228b')


# Define the company and market symbols
company_symbol = ['MBGn.DE', '7267.T', 'RACE.MI']


market_index_symbol = ['.GDAXI', '.TOPX', '.TFTIT40E']


# Define the estimation and prediction periods
estimation_start_date = '2010-01-01'
estimation_end_date = '2013-12-31'
prediction_start_date = '2014-01-01'
prediction_end_date = '2021-12-31'

# Get historical stock price data for Mercedes and the market index
mercedes_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)
market_data = ek.get_timeseries(market_index_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)

# Get future stock price data for prediction
mercedes_future_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
market_future_data = ek.get_timeseries(market_index_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)

# Calculate log returns for Mercedes and the market index
mercedes_log_returns = np.log(mercedes_data['CLOSE'] / mercedes_data['CLOSE'].shift(1))*100
market_log_returns = np.log(market_data['CLOSE'] / market_data['CLOSE'].shift(1))*100


# Drop the first row to remove NaN values
mercedes_log_returns = mercedes_log_returns.dropna()
market_log_returns = market_log_returns.dropna()

mercedes_log_returns = mercedes_log_returns.astype('float')
market_log_returns = market_log_returns.astype('float')



# Assume asset_returns and market_returns are your data
model_mercedes = arch_model(mercedes_log_returns, vol='Garch', p=1, q=1)
model_market = arch_model(market_log_returns, vol='Garch', p=1, q=1)

model_fit_mercedes = model_mercedes.fit(disp='off')
fit_market = model_market.fit(disp='off')

# Forecasted volatility (not used in the regression model as per the formula but can be used for other analyses)
forecast_vol_asset = model_fit_mercedes.forecast(start=0).variance.dropna().squeeze()
forecast_vol_market = fit_market.forecast(start=0).variance.dropna().squeeze()

# Create a new DataFrame to hold your features
data = {
    'Lagged_Asset_Return': mercedes_log_returns.shift(1),
    'Market_Return': market_log_returns
}
df = pd.DataFrame(data).dropna()  # drop the first row to remove NaN values

# Add a constant term for the intercept (alpha_0)
df = sm.add_constant(df)

# Run the regression
model = sm.OLS(mercedes_log_returns[df.index], df)  # ensure that the indices align correctly
results = model.fit()

# The alpha_0, alpha_1, and beta values will be in the results summary
print(results.summary())

# Get regression parameters into a dictionary
params_mercedes = results.params

def calculate_expected_returns(log_returns, market_returns, params):
    return params['const'] + params['Lagged_Asset_Return'] * log_returns.shift(1) + params['Market_Return'] * market_returns

# Calculate log returns for future data
mercedes_future_log_returns = np.log(mercedes_future_data['CLOSE'] / mercedes_future_data['CLOSE'].shift(1)) * 100
market_future_log_returns = np.log(market_future_data['CLOSE'] / market_future_data['CLOSE'].shift(1)) * 100

# Calculate expected returns for future data
expected_returns_mercedes = calculate_expected_returns(mercedes_future_log_returns, market_future_log_returns, params_mercedes)

# Calculate abnormal returns for future data
abnormal_returns_mercedes = mercedes_future_log_returns - expected_returns_mercedes

# Now you have the expected and abnormal returns for Mercedes from 2014 to 2021



######################### plotolás ###################################


# Dropping NaN values
mercedes_future_log_returns.dropna(inplace=True)
expected_returns_mercedes.dropna(inplace=True)
abnormal_returns_mercedes.dropna(inplace=True)



# Create a new figure
plt.figure(figsize=(12,6))

# Plot the actual returns
plt.plot(mercedes_future_log_returns.index, mercedes_future_log_returns, label='Actual Returns')

# Plot the expected returns
plt.plot(expected_returns_mercedes.index, expected_returns_mercedes, label='Expected Returns')

# Plot the abnormal returns
plt.plot(abnormal_returns_mercedes.index, abnormal_returns_mercedes, label='Abnormal Returns')

# Adding title and labels
plt.title('Mercedes Actual, Expected, and Abnormal Returns (2014-2021)')
plt.xlabel('Date')
plt.ylabel('Returns (%)')

# Adding legend
plt.legend()

# Display the plot
plt.show()




#################### tesztek ###########################

from statsmodels.tsa.stattools import adfuller

def check_stationarity(data):
    result = adfuller(data)
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is not stationary.")

check_stationarity(mercedes_log_returns)
check_stationarity(market_log_returns)


# Check residuals from the GARCH model for Mercedes
residuals_mercedes = model_fit_mercedes.resid
residuals_mercedes.plot(title='Residuals from GARCH Model for Mercedes')
plt.show()


plt.scatter(df["Market_Return"], results.resid)
plt.axhline(0, color="red")
plt.title("Residuals vs. Market Return")
plt.show()


from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

bp_test = het_breuschpagan(results.resid, results.model.exog)
if bp_test[1] > 0.05:
    print("Residuals are homoscedastic.")
else:
    print("Residuals are heteroscedastic.")

# Normality Test (Q-Q Plot)
sm.qqplot(results.resid, line='45')
plt.show()

# Shapiro-Wilk test
stat, p = shapiro(results.resid)
if p > 0.05:
    print("Residuals appear to be normally distributed.")
else:
    print("Residuals may not be normally distributed.")


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
for i, col in enumerate(df.columns):
    print(f"VIF for {col}: {vif[i]}")


# Ensure both series are of the same length
common_dates = mercedes_future_log_returns.index.intersection(expected_returns_mercedes.index)

filtered_mercedes_future_log_returns = mercedes_future_log_returns.loc[common_dates]
filtered_expected_returns_mercedes = expected_returns_mercedes.loc[common_dates]

mae = mean_absolute_error(filtered_mercedes_future_log_returns, filtered_expected_returns_mercedes)
rmse = mean_squared_error(filtered_mercedes_future_log_returns, filtered_expected_returns_mercedes, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

