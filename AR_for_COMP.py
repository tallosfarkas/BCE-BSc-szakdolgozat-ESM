# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:23:56 2023

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
from scipy import stats

# Set your app key
ek.set_app_key('37934fabc6604c9bb80b65b1e3fa6bd265ad228b')

# Define the company and market symbols
company_symbols = ['MBGn.DE', '7267.T', 'RACE.MI','ORCL.K', 'RENA.PA']
market_index_symbols = ['.GDAXI', '.TOPX', '.TFTIT40E', '.SP500', '.FCHI']

# Define the estimation and prediction periods
estimation_start_date = '2016-01-01'
estimation_end_date = '2017-12-31'
prediction_start_date = '2018-01-01'
prediction_end_date = '2023-12-05'

abnormal_returns_aggregated = pd.DataFrame()

for company_symbol, market_symbol in zip(company_symbols, market_index_symbols):
    company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)
    market_data = ek.get_timeseries(market_symbol, fields=['CLOSE'], start_date=estimation_start_date, end_date=estimation_end_date)

    company_log_returns = np.log(company_data['CLOSE'] / company_data['CLOSE'].shift(1))*100
    market_log_returns = np.log(market_data['CLOSE'] / market_data['CLOSE'].shift(1))*100

    company_log_returns = company_log_returns.dropna().astype('float')
    market_log_returns = market_log_returns.dropna().astype('float')

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

    # Getting future data
    company_future_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
    market_future_data = ek.get_timeseries(market_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)

    # Calculate log returns for future data
    company_future_log_returns = np.log(company_future_data['CLOSE'] / company_future_data['CLOSE'].shift(1)) * 100
    market_future_log_returns = np.log(market_future_data['CLOSE'] / market_future_data['CLOSE'].shift(1)) * 100

    # Calculate expected returns for future data
    expected_returns_company = params_company['const'] + params_company['Lagged_Asset_Return'] * company_future_log_returns.shift(1) + params_company['Market_Return'] * market_future_log_returns

   # Calculate abnormal returns for future data
    abnormal_returns_company = company_future_log_returns - expected_returns_company
    
    # Aggregate the abnormal returns with named columns
    abnormal_returns_aggregated[company_symbol] = abnormal_returns_company
    
    

# Now you have the abnormal returns aggregated for all the companies. You can then rank or analyze them as needed.

abnormal_returns_aggregated.dropna(inplace=True)



########################## testing ############################


# Assuming 'results' is the result of your OLS fit and 'Excess_Stock_Returns' is the dependent variable
predicted_values = OLS_results.predict(df[['const', 'Lagged_Asset_Return', 'Market_Return']])

# Plotting the actual vs. predicted values for linearity check
plt.scatter(predicted_values, df['Lagged_Asset_Return'])  # Replace 'Lagged_Asset_Return' with the correct dependent variable
plt.xlabel('Predicted Values')
plt.ylabel('Actual Returns')
plt.title('Linearity Check: Actual vs Predicted Returns')
plt.show()



# Durbin-Watson Test for Autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw_test = durbin_watson(OLS_residuals)
print("Durbin-Watson statistic:", dw_test)

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = df.columns[1:]
vif_data["VIF"] = [variance_inflation_factor(df.values, i + 1) for i in range(len(df.columns[1:]))]
print(vif_data)


# Ljung-Box Q-test on Squared Residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
ljung_box_results = acorr_ljungbox(OLS_residuals**2, lags=[10], return_df=True)
print(ljung_box_results)


from statsmodels.stats.diagnostic import het_breuschpagan

# Standardize the residuals from the GARCH model
standardized_residuals = garch_model_fit.resid / garch_model_fit.conditional_volatility

# Assuming the length of 'standardized_residuals' is the same as 'garch_model_fit.conditional_volatility'
# If not, you would need to adjust the length of 'standardized_residuals' accordingly

# Adjust 'df' to match the length of 'standardized_residuals' (assuming 'df' includes all necessary rows)
trimmed_df = df.iloc[:len(standardized_residuals)]

# Ensure the 'trimmed_df' has a constant term if your model requires it
if 'const' not in trimmed_df.columns:
    trimmed_df = sm.add_constant(trimmed_df)

# Perform the Breusch-Pagan test on the standardized residuals
# Note: The 'exog' parameter should include the same explanatory variables used in the original regression model
bp_test_garch = het_breuschpagan(standardized_residuals, trimmed_df[['const', 'Market_Return']])
print(f"Breusch-Pagan Test p-value for GARCH residuals: {bp_test_garch[1]}")






plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(OLS_residuals)
plt.title('OLS Residuals')

plt.subplot(2, 1, 2)
plt.plot(np.sqrt(garch_model_fit.conditional_volatility))
plt.title('Conditional Volatility from GARCH Model')
plt.tight_layout()
plt.show()



print("OLS Model Coefficients:\n", OLS_results.params)
print("OLS Model Coefficients p-values:\n", OLS_results.pvalues)

from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(OLS_residuals, df[['const', 'Market_Return']])
print("Breusch-Pagan Test p-value:", bp_test[1])


from scipy.stats import jarque_bera
jb_test = jarque_bera(OLS_residuals)
print("Jarque-Bera Test p-value:", jb_test[1])


############################### F1 data ###############################



import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Modify the get_race_results function to return both race date and race name
def get_race_results(year, race_number):
    url = f"http://ergast.com/api/f1/{year}/{race_number}/results.json"
    response = requests.get(url)
    data = response.json()
    
    # Extract the date and name of the race
    race_date = data['MRData']['RaceTable']['Races'][0]['date']
    race_name = data['MRData']['RaceTable']['Races'][0]['raceName']
    
    return pd.DataFrame({'Date': [race_date], 'Race Name': [race_name]})

def get_all_race_dates_and_names(start_year, end_year):
    races = []
    
    for year in range(start_year, end_year + 1):
        race_number = 1
        
        while True:
            try:
                race_results = get_race_results(year, race_number)
                race_date = race_results['Date'][0]
                race_name = race_results['Race Name'][0]
                races.append((race_date, race_name))
                race_number += 1
            except IndexError:
                break
    
    return races

all_race_dates_and_names = get_all_race_dates_and_names(2018, 2023)
all_race_dates = [date for date, _ in all_race_dates_and_names]
all_race_dates = pd.to_datetime(all_race_dates)


# Modified get_race_results function to return race date, race name, and winning team
def get_race_results(year, race_number):
    url = f"http://ergast.com/api/f1/{year}/{race_number}/results.json"
    response = requests.get(url)
    data = response.json()
    
    # Extract the date, name of the race, and winning team (constructor)
    race_info = data['MRData']['RaceTable']['Races'][0]
    race_date = race_info['date']
    race_name = race_info['raceName']
    winning_team = race_info['Results'][0]['Constructor']['name']
    
    return pd.DataFrame({'Date': [race_date], 'Race Name': [race_name], 'Winning Team': [winning_team]})

def get_all_race_dates_names_and_winners(start_year, end_year):
    races = []
    
    for year in range(start_year, end_year + 1):
        race_number = 1
        
        while True:
            try:
                race_results = get_race_results(year, race_number)
                race_date = race_results['Date'][0]
                race_name = race_results['Race Name'][0]
                winning_team = race_results['Winning Team'][0]
                races.append((race_date, race_name, winning_team))
                race_number += 1
            except IndexError:
                break
    
    return pd.DataFrame(races, columns=['Date', 'Race Name', 'Winning Team'])

# Gather data for the specified years
all_race_data = get_all_race_dates_names_and_winners(2018, 2023)

# Count the number of wins for each team
winning_teams_count = all_race_data['Winning Team'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
winning_teams_count.plot(kind='bar')
plt.title('F1 Race Winning Teams (2018 - 2023)')
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.show()

############################## AR and AAR ##############################

trading_days = abnormal_returns_aggregated.index

def find_next_trading_day(date, trading_days):
    next_day = date + pd.Timedelta(days=1)
    while next_day not in trading_days:
        next_day += pd.Timedelta(days=1)
    return next_day

# Find previous trading day before the date
def find_previous_trading_day(date, trading_days):
    previous_day = date - pd.Timedelta(days=1)
    while previous_day not in trading_days:
        previous_day -= pd.Timedelta(days=1)
    return previous_day


def calculate_ar_for_day(abnormal_returns, race_date, trading_days, offset):
    if offset > 0:
        base_day = find_next_trading_day(race_date, trading_days)
    else:
        base_day = find_previous_trading_day(race_date, trading_days)

    specific_day = base_day + pd.Timedelta(days=offset)

    # Make sure specific_day is also a trading day
    if offset > 0:
        while specific_day not in trading_days:
            specific_day += pd.Timedelta(days=1)
    else:
        while specific_day not in trading_days:
            specific_day -= pd.Timedelta(days=1)

    if specific_day in abnormal_returns.index:
        return abnormal_returns.loc[specific_day]
    else:
        return np.nan

def calculate_aar_for_day(abnormal_returns, all_race_dates, trading_days, offset):
    ars = []
    for race_date in all_race_dates:
        event_day = find_next_trading_day(race_date, trading_days)
        ar = calculate_ar_for_day(abnormal_returns, event_day, trading_days, offset)  # fixed here
        ars.append(ar)
    # Calculate AAR
    aar = np.mean(ars)
    # Calculate variance using the cross-sectional method from the image
    squared_diffs = [(ar - aar) ** 2 for ar in ars]
    variance = sum(squared_diffs) / (len(ars) - 1)
    return aar, variance

def t_test_for_aar(aar, variance, n):
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(variance)
    # Calculate the t-statistic for AAR
    t_stat = (np.sqrt(n) * aar) / std_dev
    # Calculate the p-value for the t-statistic
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
    return t_stat, p_val

# Assuming abnormal_returns_mercedes is a DataFrame or Series with dates as the index
trading_days = abnormal_returns_aggregated.index

results_all_companies = {}
for company_symbol in company_symbols:
    daily_windows = list(range(-5, 6))
    results_daily = []
    for day_offset in daily_windows:
        aar, variance = calculate_aar_for_day(abnormal_returns_aggregated[company_symbol], all_race_dates, trading_days, day_offset)
        t_stat, p_val = t_test_for_aar(aar, variance, len(all_race_dates))
        results_daily.append({
            't': day_offset,
            'AAR': aar,
            't-ratio': t_stat,
            'p-value': p_val
        })
    results_all_companies[company_symbol] = pd.DataFrame(results_daily)

# To print results for a specific company:
print(results_all_companies)



def calculate_ar_for_window(abnormal_returns, event_day, start_offset, end_offset, trading_days):
    # Adjust the start day based on the trading days
    start_day = event_day
    for _ in range(abs(start_offset)):
        start_day = find_next_trading_day(start_day, trading_days) if start_offset > 0 else find_previous_trading_day(start_day, trading_days)
    
    # Adjust the end day based on the trading days
    end_day = event_day
    for _ in range(abs(end_offset)):
        end_day = find_next_trading_day(end_day, trading_days) if end_offset > 0 else find_previous_trading_day(end_day, trading_days)

    # Ensure that the end day is not before the start day
    if end_day and start_day and end_day < start_day:
        end_day = start_day
    
    # Get the abnormal returns for the window
    ar = abnormal_returns.loc[start_day:end_day] if start_day and end_day else pd.Series(dtype='float64')
    return ar

def calculate_aar_for_window(abnormal_returns, all_race_dates, trading_days, start_offset, end_offset):
    ars = []
    for race_date in all_race_dates:
        # Convert race_date to the actual event day (next trading day)
        event_day = find_next_trading_day(pd.to_datetime(race_date), trading_days)
        if event_day is not None:  # Make sure the event day is valid
            ar_window = calculate_ar_for_window(abnormal_returns, event_day, start_offset, end_offset, trading_days)
            ars.extend(ar_window)  # Use extend instead of append to flatten the list

    # Calculate AAR for the window
    aar = np.nanmean(ars)  # Use nanmean to ignore NaN values
    
    # Calculate variance using the cross-sectional method
    squared_diffs = [(ar - aar) ** 2 for ar in ars if not np.isnan(ar)]
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)
    
    return aar, variance

def t_test_for_aar(aar, variance, n):
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(variance)
    # Calculate the t-statistic for AAR using the cross-sectional method
    t_stat = (np.sqrt(n) * aar) / std_dev
    # Calculate the p-value for the t-statistic using a two-tailed test
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_val



results_all_companies = {}
for company_symbol in company_symbols:
    trading_days = abnormal_returns_aggregated.index
    extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]
    results_extended = []
    for window in extended_windows:
        start, end = window
        aar, variance = calculate_aar_for_window(abnormal_returns_aggregated[company_symbol], all_race_dates, trading_days, start, end)
        t_stat, p_val = t_test_for_aar(aar, variance, len(all_race_dates))
        results_extended.append({
            'Window': window,
            'AAR': aar,
            'Variance': variance,
            't-statistic': t_stat,
            'p-value': p_val
        })
    results_all_companies[company_symbol] = pd.DataFrame(results_extended)

for company_symbol, results_df in results_all_companies.items():
    print(f"Results for {company_symbol}:\n", results_df)


################################### CAR CAAR ###########################################


trading_days = abnormal_returns_aggregated.index

def find_next_trading_day(date, trading_days):
    next_day = date + pd.Timedelta(days=1)
    while next_day not in trading_days:
        next_day += pd.Timedelta(days=1)
    return next_day

# Find previous trading day before the date
def find_previous_trading_day(date, trading_days):
    previous_day = date - pd.Timedelta(days=1)
    while previous_day not in trading_days:
        previous_day -= pd.Timedelta(days=1)
    return previous_day

def calculate_car_for_window(abnormal_returns, event_day, start_offset, end_offset, trading_days):
    # Find the actual start day and end day based on trading days
    if start_offset < 0:  # If we're looking before the event
        start_day = find_previous_trading_day(event_day, trading_days)
        for _ in range(abs(start_offset) - 1):
            start_day = find_previous_trading_day(start_day, trading_days)
    else:  # If we're starting from the event or after
        start_day = event_day
        for _ in range(start_offset):
            start_day = find_next_trading_day(start_day, trading_days)

    if end_offset > 0:  # If we're looking after the event
        end_day = find_next_trading_day(event_day, trading_days)
        for _ in range(end_offset - 1):
            end_day = find_next_trading_day(end_day, trading_days)
    else:  # If the end day is the event day itself
        end_day = event_day

    # Sum abnormal returns within the window
    car = abnormal_returns.loc[start_day:end_day].sum()
    return car

def calculate_caar_for_window(abnormal_returns, all_race_dates, trading_days, start_offset, end_offset):
    cars = []
    for race_date in all_race_dates:
        # Convert race_date to the actual event day (next trading day)
        event_day = find_next_trading_day(pd.to_datetime(race_date), trading_days)
        if event_day is not None:  # Make sure the event day is valid
            car = calculate_car_for_window(abnormal_returns, event_day, start_offset, end_offset, trading_days)
            cars.append(car)

    # Calculate CAAR
    caar = np.nanmean(cars)  # Use nanmean to ignore NaN values
    # Calculate variance using the cross-sectional method from the image
    squared_diffs = [(car - caar) ** 2 for car in cars if not np.isnan(car)]
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)
    return caar, variance

def t_test_for_caar(caar, variance, n):
    # Calculate the standard deviation from the variance
    std_dev = np.sqrt(variance)
    # Calculate the t-statistic for CAAR
    t_stat = (np.sqrt(n) * caar) / std_dev
    # Calculate the two-tailed p-value for the t-statistic
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
    return t_stat, p_val


results_all_companies = {}
extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]

for company_symbol in company_symbols:
    trading_days = abnormal_returns_aggregated[company_symbol].index
    results_extended = []
    for window in extended_windows:
        start, end = window
        caar, variance = calculate_caar_for_window(abnormal_returns_aggregated[company_symbol], all_race_dates, trading_days, start, end)
        t_stat, p_val = t_test_for_caar(caar, variance, len(all_race_dates))
        results_extended.append({
            'Window': window,
            'CAAR': caar,
            'Variance': variance,
            't-statistic': t_stat,
            'p-value': p_val
        })
    results_all_companies[company_symbol] = pd.DataFrame(results_extended)

for company_symbol, results_df in results_all_companies.items():
    print(f"Results for {company_symbol}:\n", results_df)


############################### accross teams #######################################

# Average AR and AAR across all companies

# Create a DataFrame that averages abnormal returns across companies for each trading day
avg_abnormal_returns = abnormal_returns_aggregated.mean(axis=1)

daily_windows = list(range(-5, 6))
results_daily = []
for day_offset in daily_windows:
    aar, variance = calculate_aar_for_day(avg_abnormal_returns, all_race_dates, trading_days, day_offset)
    t_stat, p_val = t_test_for_aar(aar, variance, len(all_race_dates))
    results_daily.append({
        't': day_offset,
        'AAR': aar,
        't-ratio': t_stat,
        'p-value': p_val
    })
print(pd.DataFrame(results_daily))

# Average CAR and CAAR across all companies

# Create a DataFrame that contains the cumulative abnormal returns for each company
cumulative_abnormal_returns = abnormal_returns_aggregated.cumsum()

# Calculate CAAR for the specified windows
results_extended = []
for window in extended_windows:
    start, end = window
    caar, variance = calculate_caar_for_window(avg_abnormal_returns, all_race_dates, trading_days, start, end)
    t_stat, p_val = t_test_for_caar(caar, variance, len(all_race_dates))
    results_extended.append({
        'Window': window,
        'CAAR': caar,
        'Variance': variance,
        't-statistic': t_stat,
        'p-value': p_val
    })
print(pd.DataFrame(results_extended))




def find_next_trading_day(date, trading_days):
    next_day = date + pd.Timedelta(days=1)
    while next_day not in trading_days:
        next_day += pd.Timedelta(days=1)
    return next_day

def find_previous_trading_day(date, trading_days):
    previous_day = date - pd.Timedelta(days=1)
    while previous_day not in trading_days:
        previous_day -= pd.Timedelta(days=1)
    return previous_day

def calculate_ar_for_window(abnormal_returns, event_day, start_offset, end_offset, trading_days):
    start_day = event_day
    for _ in range(abs(start_offset)):
        start_day = find_next_trading_day(start_day, trading_days) if start_offset > 0 else find_previous_trading_day(start_day, trading_days)
        
    end_day = start_day
    for _ in range(abs(start_offset), abs(end_offset) + 1):
        end_day = find_next_trading_day(end_day, trading_days) if end_offset > 0 else find_previous_trading_day(end_day, trading_days)
    
    # Ensure that the end day is not before the start day
    if end_day < start_day:
        end_day = start_day
    
    # Get the abnormal returns for the window
    ar_window = abnormal_returns.loc[start_day:end_day]
    return ar_window

def calculate_aar_for_window(aggregated_abnormal_returns, all_race_dates, trading_days, start_offset, end_offset):
    aar_list = []
    ars_list = []
    for race_date in all_race_dates:
        event_day = find_next_trading_day(pd.to_datetime(race_date), trading_days)
        if event_day is not None:
            ar_window = calculate_ar_for_window(aggregated_abnormal_returns, event_day, start_offset, end_offset, trading_days)
            ars_list.extend(ar_window.tolist())  # Collect all ARs across all windows
            aar_list.append(ar_window.mean())  # Calculate AAR for each window and append

    # Calculate the mean of AARs over all windows
    aar = np.mean(aar_list)
    # Calculate the cross-sectional variance of ARs
    variance = np.sum((np.array(ars_list) - aar) ** 2) / (len(ars_list) - 1)
    return aar, variance

def t_test_for_aar(aar, variance, n):
    std_dev = np.sqrt(variance)
    t_stat = (np.sqrt(n) * aar) / std_dev
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-1))
    return t_stat, p_val

# Assuming 'abnormal_returns_aggregated' is a DataFrame with the abnormal returns of companies as columns and dates as the index
# First, aggregate the abnormal returns across all companies for each event date
aggregated_abnormal_returns = abnormal_returns_aggregated.mean(axis=1)

# Assuming 'all_race_dates' is a list of dates when events occurred
# And 'trading_days' is a pandas Index of all trading days
results_aggregated = []
extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]

for window in extended_windows:
    start_offset, end_offset = window
    aar, variance = calculate_aar_for_window(abnormal_returns_aggregated.mean(axis=1), all_race_dates, aggregated_abnormal_returns.index, start_offset, end_offset)
    t_stat, p_val = t_test_for_aar(aar, variance, len(all_race_dates))
    results_aggregated.append({
        'Window': window,
        'AAR': aar,
        'Variance': variance,
        't-statistic': t_stat,
        'p-value': p_val
    })

# Convert the results to a DataFrame for easier analysis
results_df_aggregated = pd.DataFrame(results_aggregated)
print("Aggregated Results across all companies:\n", results_df_aggregated)


################################### plot ##################################
# Plot for Abnormal Returns
plt.figure(figsize=(16, 10))
for company in abnormal_returns_aggregated.columns:
    plt.plot(abnormal_returns_aggregated.index, abnormal_returns_aggregated[company], label=company)

plt.title('Abnormal Returns for Each Company')
plt.xlabel('Date')
plt.ylabel('Abnormal Returns (%)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot for Cumulative Abnormal Returns
cumulative_abnormal_returns = abnormal_returns_aggregated.cumsum()

plt.figure(figsize=(16, 10))
for company in cumulative_abnormal_returns.columns:
    plt.plot(cumulative_abnormal_returns.index, cumulative_abnormal_returns[company], label=company)

plt.title('Cumulative Abnormal Returns for Each Company')
plt.xlabel('Date')
plt.ylabel('Cumulative Abnormal Returns (%)')
plt.legend()
plt.tight_layout()
plt.show()




# Define the company symbols
company_symbols = ['MBGn.DE', '7267.T', 'RACE.MI', 'ORCL.K', 'RENA.PA']

# Define the estimation and prediction periods
estimation_start_date = '2016-01-01'
estimation_end_date = '2017-12-31'
prediction_start_date = '2018-01-01'
prediction_end_date = '2023-12-05'

# Initialize a DataFrame to store cumulative log returns
cumulative_log_returns = pd.DataFrame()

# Fetch the data and calculate cumulative log returns for each company
for company_symbol in company_symbols:
    company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
    
    # Calculate log returns
    company_log_returns = np.log(company_data['CLOSE'] / company_data['CLOSE'].shift(1))
    
    # Calculate cumulative log returns
    company_cumulative_log_returns = company_log_returns.cumsum()
    
    # Store the cumulative log returns in the DataFrame
    cumulative_log_returns[company_symbol] = company_cumulative_log_returns

# Drop the NaN values from the cumulative log returns
cumulative_log_returns.dropna(inplace=True)

# Plot the cumulative log returns for each company
plt.figure(figsize=(16, 10))
for company_symbol in company_symbols:
    plt.plot(cumulative_log_returns.index, cumulative_log_returns[company_symbol], label=company_symbol)

plt.title('A vizsgált vállalatok kumulált loghozamai',fontsize=20, weight='bold')
plt.xlabel('Dátum',fontsize=20)
plt.ylabel('Loghozamok',fontsize=20)
plt.grid(True)
plt.legend()
plt.show()






# Define the company symbols
company_symbols = ['MBGn.DE', '7267.T', 'RACE.MI', 'ORCL.K', 'RENA.PA']

# Define the estimation and prediction periods
estimation_start_date = '2016-01-01'
estimation_end_date = '2017-12-31'
prediction_start_date = '2018-01-01'
prediction_end_date = '2023-12-05'

# Initialize DataFrames to store cumulative log returns and abnormal returns
cumulative_log_returns = pd.DataFrame()
cumulative_abnormal_returns = pd.DataFrame()

# Fetch the data and calculate cumulative log returns for each company
for company_symbol in company_symbols:
    company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
    
    # Calculate log returns
    company_log_returns = np.log(company_data['CLOSE'] / company_data['CLOSE'].shift(1))
    
    # Calculate cumulative log returns
    company_cumulative_log_returns = company_log_returns.cumsum()
    
    # Store the cumulative log returns in the DataFrame
    cumulative_log_returns[company_symbol] = company_cumulative_log_returns

# Drop the NaN values from the cumulative log returns
cumulative_log_returns.dropna(inplace=True)

# Calculate cumulative abnormal returns
cumulative_abnormal_returns = cumulative_log_returns - cumulative_log_returns.mean()

# Plot the cumulative log returns and cumulative abnormal returns for each company
plt.figure(figsize=(16, 10))

# Iterate over each company
for company_symbol in company_symbols:
    # Plot cumulative log returns
    plt.plot(cumulative_log_returns.index, cumulative_log_returns[company_symbol], label=f'{company_symbol} Log Returns')
    
    # Plot cumulative abnormal returns with a different style
    plt.plot(cumulative_abnormal_returns.index, cumulative_abnormal_returns[company_symbol], linestyle='--', label=f'{company_symbol} Abnormal Returns')

plt.title('Cumulative Log and Abnormal Returns for Companies')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()













company_symbols = ['MBGn.DE', '7267.T', 'RACE.MI', 'ORCL.K', 'RENA.PA']

# Define the estimation and prediction periods
estimation_start_date = '2016-01-01'
estimation_end_date = '2017-12-31'
prediction_start_date = '2018-01-01'
prediction_end_date = '2023-12-05'

# Initialize a DataFrame to store stock prices
stock_prices = pd.DataFrame()

# Fetch the closing stock prices for each company
for company_symbol in company_symbols:
    company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
    
    # Store the stock prices in the DataFrame
    stock_prices[company_symbol] = company_data['CLOSE']

# Drop the NaN values from the stock prices
stock_prices.dropna(inplace=True)

# Plot the stock prices for each company
plt.figure(figsize=(16, 10))

# Iterate over each company
for company_symbol in company_symbols:
    plt.plot(stock_prices.index, stock_prices[company_symbol], label=company_symbol)

plt.title('Stock Prices for Companies')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






# Define the company symbols
company_symbols = ['MBGn.DE', '7267.T', 'RACE.MI', 'ORCL.K', 'RENA.PA']

# Define a color palette
color_palette = {
    'MBGn.DE': 'royalblue',
    '7267.T': 'forestgreen',
    'RACE.MI': 'crimson',
    'ORCL.K': 'goldenrod',
    'RENA.PA': 'purple'
}

# Define the estimation and prediction periods
estimation_start_date = '2016-01-01'
estimation_end_date = '2017-12-31'
prediction_start_date = '2018-01-01'
prediction_end_date = '2023-12-05'

# Initialize a DataFrame to store stock prices
stock_prices = pd.DataFrame()

# Fetch the closing stock prices for each company
for company_symbol in company_symbols:
    company_data = ek.get_timeseries(company_symbol, fields=['CLOSE'], start_date=prediction_start_date, end_date=prediction_end_date)
    
    # Store the stock prices in the DataFrame
    stock_prices[company_symbol] = company_data['CLOSE']

# Drop the NaN values from the stock prices
stock_prices.dropna(inplace=True)

# Set the number of rows for the subplots based on the number of companies
num_rows = len(company_symbols)

# Create subplots
plt.figure(figsize=(16, 4 * num_rows))

# Iterate over each company to create a subplot
for i, company_symbol in enumerate(company_symbols, 1):
    plt.subplot(num_rows, 1, i)
    plt.plot(stock_prices.index, stock_prices[company_symbol], label=company_symbol, color=color_palette[company_symbol])
    plt.title(f'A {company_symbol} részvényárfolyama',fontsize=24, weight='bold')
    plt.xlabel('Dátum',fontsize=20)
    plt.ylabel('Árfolyam',fontsize=20)
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()











# Gather data for the specified years
all_race_data = get_all_race_dates_names_and_winners(2018, 2023)

# Convert the 'Date' column to datetime
all_race_data['Date'] = pd.to_datetime(all_race_data['Date'])

# Sort the races by date
all_race_data.sort_values('Date', inplace=True)

# Create a list of all teams
teams = all_race_data['Winning Team'].unique()

# Initialize a DataFrame to track cumulative wins
cumulative_wins = pd.DataFrame(0, index=all_race_data['Date'].drop_duplicates(), columns=teams)

# Populate the cumulative wins DataFrame
for index, row in all_race_data.iterrows():
    # Increment the win count for the winning team on the race date
    cumulative_wins.at[row['Date'], row['Winning Team']] += 1

# Calculate the cumulative sum of the wins for each team
cumulative_wins = cumulative_wins.cumsum()


# Find the first date of the year from your race dates
first_date_of_year = pd.Timestamp(year=2018, month=1, day=1)

# Add a row with zeros for all teams at the beginning of your DataFrame
# This is done by creating a new DataFrame with the first row and appending your existing DataFrame to it
first_row = pd.DataFrame(0, index=[first_date_of_year], columns=cumulative_wins.columns)
cumulative_wins = pd.concat([first_row, cumulative_wins]).fillna(0)

# Ensure the DataFrame is sorted by date
cumulative_wins.sort_index(inplace=True)
cumulative_wins = cumulative_wins.drop(['AlphaTauri', 'Racing Point', 'McLaren'], axis=1)

plt.figure(figsize=(16, 10))
for constructor in cumulative_wins.columns:
    plt.step(cumulative_wins.index, cumulative_wins[constructor], label=constructor, where='post', linewidth=3)

plt.title('Kumulált győzelmek az F1-es csapatok között (2018 - 2023)',fontsize=20, weight='bold')
plt.xlabel('Dátum',fontsize=15, weight='bold')
plt.ylabel('Kumulált győzelmek',fontsize=15, weight='bold')
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()






# Group by year and reset cumulative count for each year
cumulative_wins_by_year = {
    year: data.sub(data.iloc[0]).astype(int)
    for year, data in cumulative_wins.groupby(cumulative_wins.index.year)
}

# Create subplots for each year
fig, axs = plt.subplots(nrows=len(cumulative_wins_by_year), ncols=1, figsize=(10, 6 * len(cumulative_wins_by_year)))

# Check if there's only one year and adjust the axs array
if len(cumulative_wins_by_year) == 1:
    axs = [axs]

# Plot data for each year
for ax, (year, data) in zip(axs, cumulative_wins_by_year.items()):
    for constructor in data.columns:
        ax.step(data.index, data[constructor], label=constructor, where='post', linewidth=2)
    ax.set_title(f'Cumulative Wins in {year}', fontsize=14, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Wins', fontsize=12)
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()






# Group by year and reset cumulative count for each year
cumulative_wins_by_year = {
    year: data.sub(data.iloc[0]).astype(int)
    for year, data in cumulative_wins.groupby(cumulative_wins.index.year)
}

# Calculate the number of rows and columns for subplots
num_years = len(cumulative_wins_by_year)
num_cols = 2  # Two subplots per row
num_rows = -(-num_years // num_cols)  # Ceiling division to get the number of rows

# Create subplots with 2 columns
fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, num_rows * 5))

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Plot data for each year
for ax, (year, data) in zip(axs, cumulative_wins_by_year.items()):
    for constructor in data.columns:
        ax.step(data.index, data[constructor], label=constructor, where='post', linewidth=2)
    ax.set_title(f'Cumulative Wins in {year}', fontsize=14, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Wins', fontsize=12)
    ax.legend()
    ax.grid(True)

# Hide any unused subplots
for i in range(len(cumulative_wins_by_year), len(axs)):
    fig.delaxes(axs[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()










import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Assuming 'cumulative_wins_by_year' is a dictionary with years as keys and DataFrames as values

# Calculate the number of rows and columns for subplots
num_years = len(cumulative_wins_by_year)
num_cols = 2  # We'll have two subplots per row
num_rows = -(-num_years // num_cols)  # Ceiling division to get the number of rows

# Create a figure with GridSpec for flexible subplot layout
fig = plt.figure(figsize=(15, num_rows * 5))
gs = GridSpec(num_rows, num_cols, figure=fig)

# Track the current row and column for placing subplots
current_row = 0
current_col = 0

# Plot data for each year
for year, data in cumulative_wins_by_year.items():
    # If it's the last plot and we have an odd number of plots, create a centered subplot
    if year == max(cumulative_wins_by_year.keys()) and num_years % num_cols != 0:
        ax = fig.add_subplot(gs[current_row, :])
    else:
        ax = fig.add_subplot(gs[current_row, current_col])

    # Plot the cumulative wins for each constructor
    for constructor in data.columns:
        ax.step(data.index, data[constructor], label=constructor, where='post', linewidth=2)

    ax.set_title(f'Kumulált győzelmek {year}', fontsize=14, weight='bold')
    ax.set_xlabel('Dátum', fontsize=12)
    ax.set_ylabel('Kumulált Győzelmek', fontsize=12)
    ax.legend()
    ax.grid(True)

    # Update the column; if it's the end of a row, reset to the first column and go to the next row
    current_col += 1
    if current_col >= num_cols:
        current_col = 0
        current_row += 1

# If we had an odd number of years, the last subplot is already centered. Adjust the layout.
plt.tight_layout()

# Show the plot
plt.show()
