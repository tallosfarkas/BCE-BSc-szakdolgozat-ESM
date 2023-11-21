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
company_symbols = ['7267.T', 'ORCL.K']
market_index_symbols = [ '.TOPX', '.SP500']







# Define the estimation and prediction periods
estimation_start_date = '2022-01-01'
estimation_end_date = '2022-12-31'
prediction_start_date = '2023-01-01'
prediction_end_date = '2023-12-31'

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

all_race_dates_and_names = get_all_race_dates_and_names(2023, 2023)
all_race_dates = [date for date, _ in all_race_dates_and_names]
all_race_dates = pd.to_datetime(all_race_dates)

all_race_dates_and_names = all_race_dates_and_names[:14]
all_race_dates = all_race_dates[:14]
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


aggregated_abnormal_returns = abnormal_returns_aggregated.mean(axis=1)
# Calculate the aggregated AAR for all companies across different windows
trading_days = abnormal_returns_aggregated.index
extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]
results_aggregated = []

# Aggregate the abnormal returns for all companies
aggregated_abnormal_returns = abnormal_returns_aggregated.sum(axis=1)

for window in extended_windows:
    start, end = window
    aar, variance = calculate_aar_for_window(aggregated_abnormal_returns, all_race_dates, trading_days, start, end)
    t_stat, p_val = t_test_for_caar(aar, variance, len(all_race_dates))
    results_aggregated.append({
        'Window': window,
        'AAR': aar,
        'Variance': variance,
        't-statistic': t_stat,
        'p-value': p_val
    })

results_df_aggregated = pd.DataFrame(results_aggregated)
print("Aggregated Results across all companies:\n", results_df_aggregated)

