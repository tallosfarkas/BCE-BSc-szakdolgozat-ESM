# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:51:42 2023

@author: fajka
"""

import eikon as ek
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
import arch
from arch import arch_model

# Set your app key
ek.set_app_key('37934fabc6604c9bb80b65b1e3fa6bd265ad228b')



# Define the company and market symbols
company_symbol = 'VOWG.DE'
market_index_symbol = '.GDAXI'



event_date = pd.Timestamp('2022-08-26')

# 5 trading days before the event
event_window_start_date = (event_date - BDay(7)).strftime('%Y-%m-%d')

# 5 trading days after the event
event_window_end_date = (event_date + BDay(5)).strftime('%Y-%m-%d')

# One trading year (252 days) before the event window starts
estimation_start_date = (event_date - BDay(257)).strftime('%Y-%m-%d')  # 252 + 5
estimation_end_date = (event_date - BDay(6)).strftime('%Y-%m-%d')  # One day before event window start date



# Get historical stock price data for the company and the market index
company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)
market_data = ek.get_timeseries(market_index_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)

# Get future stock price data for prediction
company_future_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=event_window_start_date, end_date=event_window_end_date)
market_future_data = ek.get_timeseries(market_index_symbol, fields=['CLOSE'], start_date=event_window_start_date, end_date=event_window_end_date)

# Calculate log returns for the company and the market index
company_log_returns = np.log(company_data['CLOSE'] / company_data['CLOSE'].shift(1))*100
market_log_returns = np.log(market_data['CLOSE'] / market_data['CLOSE'].shift(1))*100


# Drop the first row to remove NaN values
company_log_returns = company_log_returns.dropna()
market_log_returns = market_log_returns.dropna()

company_log_returns = company_log_returns.astype('float')
market_log_returns = market_log_returns.astype('float')






data = {
 'Lagged_Asset_Return': company_log_returns.shift(1),
 'Market_Return': market_log_returns
}
df = pd.DataFrame(data).dropna()
 
# Add a constant to the model
df = sm.add_constant(df)

 # Step 2: Run OLS regression
OLS_model = sm.OLS(company_log_returns[df.index], df)
OLS_results = OLS_model.fit()
 
 # Step 3: Extract residuals
OLS_residuals = OLS_results.resid
 
 # Step 4: Fit a GARCH model on the OLS residuals
garch_model = arch_model(OLS_residuals, vol='Garch', p=1, q=1)
garch_model_fit = garch_model.fit(disp='off')
 
 
garch_results = garch_model.fit()
 
 
params_company = OLS_results.params



 # Calculate log returns for future data
company_future_log_returns = np.log(company_future_data['CLOSE'] / company_future_data['CLOSE'].shift(1)) * 100
market_future_log_returns = np.log(market_future_data['CLOSE'] / market_future_data['CLOSE'].shift(1)) * 100

 # Calculate expected returns for future data
expected_returns_company = params_company['const'] + params_company['Lagged_Asset_Return'] * company_future_log_returns.shift(1) + params_company['Market_Return'] * market_future_log_returns

# Calculate abnormal returns for future data
abnormal_returns_company = company_future_log_returns - expected_returns_company

# Now you have the expected and abnormal returns for Mercedes from 2014 to 2021









# Assuming 'abnormal_returns_volkswagen' is a pandas Series with a DateTime index and 'cumulative_abnormal_returns' is calculated
abnormal_returns_volkswagen = abnormal_returns_company.dropna()

# Cumulative Abnormal Returns (CAR)
cumulative_abnormal_returns = abnormal_returns_volkswagen.cumsum()
car_volkswagen = cumulative_abnormal_returns

company_symbol = 'VOWG.DE'
event_date = pd.Timestamp('2022-08-26')

# Plotting
plt.figure(figsize=(12,6))
plt.plot(cumulative_abnormal_returns.index, cumulative_abnormal_returns, label='Kumulatív abnormális hozamok', color="#008CC8", linewidth=2)
plt.axvline(x=event_date, color='gray', linestyle='--', label='A bejelentés napja', linewidth=2)

# Highlighting the event window
plt.axvspan(event_date, event_date, color='lightgray', alpha=0.3)

plt.title(f'Kumulált abnormális hozamok a Volkswagen csoport részvényárfolyamaira vonatkozóan \naz Audi F1-be történő belépésének bejelentése körül', fontsize=14, weight='bold')
plt.xlabel('Dátum', fontsize=12)
plt.ylabel('CAR (%)', fontsize=12)
plt.legend()

# Beautifying the plot
plt.grid(True, which='major', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

# Rotate date labels
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust the padding between and around subplots.

# Display the plot
plt.show()








# Import the necessary library for t-test
from scipy import stats

# Calculate the standard deviation of ARs
std_AR = abnormal_returns_volkswagen.std()

# Display Abnormal Returns (ARs) for each day with t-statistic and p-value
print("Day\tAR\tt-stat\tp-value")
for day, ar in abnormal_returns_volkswagen.items():
    # Calculate the t-statistic for the given day's AR
    t_stat_day = ar / std_AR
    
    # Calculate the degrees of freedom
    df = len(abnormal_returns_volkswagen) - 1
    
    # Calculate the two-tailed p-value for the given day's AR
    p_val_day = 2 * (1 - stats.t.cdf(np.abs(t_stat_day), df=df))
    
    # Printing the day, AR, t-statistic, and p-value
    print(f"{day.strftime('%Y-%m-%d')}\t{ar:.2f}%\t{t_stat_day:.4f}\t{p_val_day:.4f}")

# Plotting ARs for each day
# Sample data for plotting
company_symbol = 'VOWG.DE'

darker_gray = ("#00A19B")
# Plotting ARs for each day
plt.figure(figsize=(12, 6))
bars = plt.bar(range(-5, 6), abnormal_returns_volkswagen, color='lightgray')  # Using a soft blue color for the bars

# Highlighting the event date bar
bars[5].set_color("#008CC8")  # Change color of the event date bar

# Adding a red vertical line to indicate the event date
plt.axvline(x=0, color='lightgray', linestyle='--', label='A bejelentés napja', linewidth=2)

# Adding a title and labels
plt.title('Abnormális hozamok a Volkswagen csoport részvényárfolyamaira vonatkozóan \naz Audi F1-be történő belépésének bejelentése körül', fontsize=14, weight='bold')
plt.xlabel('Kereskedési napok az eseményhez képest', fontsize=12)
plt.xticks(range(-5, 6))
plt.ylabel('AR (%)', fontsize=12)

# Adding a legend with a shadow
plt.legend(shadow=True)

# Making all spines visible for a full edge
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# You can adjust the color and width of the spines if needed
for spine in plt.gca().spines.values():
    spine.set_color('black')
    spine.set_linewidth(1)

# Adding a light grid for better readability
plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

# Adjusting the margins and layout
plt.tight_layout()

# Display the plot
plt.show()

# Display CAR at the end of the event window
print(f"Cumulative Abnormal Return (CAR) on the last event day for {company_symbol} is {car_volkswagen.iloc[-1]:.2f}%")
from scipy import stats

abnormal_returns_volkswagen.dropna()
abnormal_returns_volkswagen = abnormal_returns_volkswagen.astype(float)
# Calculate t-test for Abnormal Returns
t_stat_AR, p_val_AR = stats.ttest_1samp(abnormal_returns_volkswagen, 0)

print(f"t-statistic for AR: {t_stat_AR:.4f}")
print(f"p-value for AR: {p_val_AR:.4f}")


# Calculate the standard error for CAR
std_err_CAR = np.sqrt(len(abnormal_returns_volkswagen)) * abnormal_returns_volkswagen.std()

# Calculate the t-statistic for CAR
t_stat_CAR = car_volkswagen.iloc[-1] / std_err_CAR

# Calculate the two-tailed p-value for CAR
p_val_CAR = 2 * (1 - stats.t.cdf(np.abs(t_stat_CAR), df=len(abnormal_returns_volkswagen)-1))

print(f"t-statistic for CAR: {t_stat_CAR:.4f}")
print(f"p-value for CAR: {p_val_CAR:.4f}")


