import os
import requests
import pandas as pd
import numpy as np
from scipy.stats import t
from pandas.tseries.offsets import BDay

# Define the directory where the option data files are stored
directory = "C:/Users/fajka/OneDrive - Corvinus University of Budapest/7. SZAKDOGA/Adatok/options"
files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

frames = []

for file in files:
    filepath = os.path.join(directory, file)
    df = pd.read_excel(filepath)
    # Drop the first row and reset the index
    df = df.drop(index=df.index[0]).reset_index(drop=True)
    # Get the symbol from the filename (without the .xlsx extension)
    symbol = file[:-5]
    # Rename the columns with the symbol as prefix
    df.columns = [f"{symbol}_Date", f"{symbol}_Volume", f"{symbol}_OpenInt"]
    # Ensure the 'Date' column is in datetime format
    df[f"{symbol}_Date"] = pd.to_datetime(df[f"{symbol}_Date"])
    # Set the 'Date' column as the index
    df.set_index(f"{symbol}_Date", inplace=True)
    frames.append(df)

# Concatenate all DataFrames along the columns
combined_df = pd.concat(frames, axis=1)

# Drop all rows where any element is NaN across all DataFrames
combined_df.dropna(how='any', inplace=True)

def get_race_results(year, race_number):
    url = f"http://ergast.com/api/f1/{year}/{race_number}/results.json"
    response = requests.get(url)
    data = response.json()
    
    # Extract the date and name of the race
    # Check if the race data is present
    if data['MRData']['RaceTable']['Races']:
        race_date = data['MRData']['RaceTable']['Races'][0]['date']
        race_name = data['MRData']['RaceTable']['Races'][0]['raceName']
        return pd.DataFrame({'Date': [race_date], 'Race Name': [race_name]})
    else:
        raise IndexError('No race data found')

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
                # No more races found for this year, break out of the loop
                break
    
    return races

# Use the function to get all race dates and names from 2021 to 2023
all_race_dates_and_names = get_all_race_dates_and_names(2021, 2023)

# Convert the list of tuples to a DataFrame
race_dates_df = pd.DataFrame(all_race_dates_and_names, columns=['Date', 'Race Name'])

# Convert the date strings to datetime objects
race_dates_df['Date'] = pd.to_datetime(race_dates_df['Date'])

# Remove the last two races
race_dates_df = race_dates_df.iloc[:-2]
# Ensure race dates are in datetime format




# The column names should be checked to ensure they follow the expected pattern
# For instance, the volume columns should be named like 'MERCEDES_Volume'
volume_columns = [col for col in combined_df.columns if 'Volume' in col]
openint_columns = [col for col in combined_df.columns if 'OpenInt' in col]

# Make sure that both lists have the same length and corresponding elements
assert len(volume_columns) == len(openint_columns)

# Create log volume and abnormal volume columns for the estimation period.
for volume_col, openint_col in zip(volume_columns, openint_columns):
    symbol = volume_col.split('_')[0]
    log_volume_col = f'{symbol}_log_volume'
    mean_volume_col = f'{symbol}_mean_volume'
    abnormal_volume_col = f'{symbol}_abnormal_volume'
    
    # Make sure you only perform log calculations on positive volumes
    combined_df[volume_col] = combined_df[volume_col].apply(lambda x: x if x > 0 else np.nan)
    combined_df[openint_col] = combined_df[openint_col].apply(lambda x: x if x > 0 else np.nan)
    combined_df.dropna(subset=[volume_col, openint_col], inplace=True)
    
    combined_df[log_volume_col] = np.log((combined_df[volume_col]/combined_df[openint_col]) * 100 + 0.000255)
    
    # Now define the estimation period
    estimation_period = combined_df['2020-01-01':'2020-12-31']
    
    # Calculate the mean log volume over the estimation period for each symbol
    mean_log_volume = estimation_period[log_volume_col].mean()
    
    # Store the mean volume back in the combined_df for each symbol
    combined_df[mean_volume_col] = mean_log_volume
    
    # Calculate abnormal volume for each symbol
    combined_df[abnormal_volume_col] = combined_df[log_volume_col] - combined_df[mean_volume_col]



# Define T as the number of days in the estimation period
T = estimation_period.shape[0]

# Calculate the mean of the mean-adjusted abnormal trading volumes over the estimation period
mean_adjusted_abnormal_volumes = combined_df[[col for col in combined_df.columns if 'abnormal_volume' in col]].mean()
mean_of_means = mean_adjusted_abnormal_volumes.mean()

# Calculate the standard deviation of the mean-adjusted abnormal trading volumes over the estimation period
std_dev = np.sqrt(((mean_adjusted_abnormal_volumes - mean_of_means) ** 2).sum() / (T - 1))

# Calculate event study metrics for each race
portfolio_results = []
company_results = []

for race_date in race_dates_df['Date']:
    
    # Adjust for the next trading day after the race
    event_date = race_date + BDay(1)
    
    # Define the event window
    event_start = race_date - BDay(5)
    event_end = event_date + BDay(5)
    
    
    # Check if event_date is in the index of combined_df
    if event_date not in combined_df.index:
        # If not, find the next trading day that is in the index
        next_trading_days = combined_df[combined_df.index > event_date].index
        if not next_trading_days.empty:
            event_date = next_trading_days[0]
        else:
            # If there are no trading days after event_date, skip this date
            print(f"No trading days available after {race_date}. Skipping.")
            continue
    
    
    
    # Get the event window data
    event_window = combined_df.loc[event_start:event_end]
    
    # Get the event window data
    event_window = combined_df.loc[event_start:event_end]
    
    # Calculate equal-weighted portfolio mean abnormal trading volume on the event date
    abnormal_volume_cols = [col for col in combined_df.columns if 'abnormal_volume' in col]
    event_abnormal_volumes = event_window[abnormal_volume_cols].mean(axis=1)
    
    # Try to access the event_date data
    try:
        portfolio_mean_volume = event_abnormal_volumes.loc[event_date]
    except KeyError:
        print(f"Date {event_date} not found in event window. Skipping.")
        continue
    
    # Use the previously calculated standard deviation to calculate the test statistic for the event date
    portfolio_mean_volume = event_abnormal_volumes.loc[event_date]
    test_statistic = portfolio_mean_volume / std_dev
    degrees_of_freedom = T - 1
    
    # Calculate individual company abnormal volumes and their test statistics
    for col in abnormal_volume_cols:
        company_symbol = col.split('_')[0]
        company_abnormal_volume = event_window[col].loc[event_date]
        company_test_statistic = company_abnormal_volume / std_dev
        company_p_value = (1 - t.cdf(abs(company_test_statistic), df=degrees_of_freedom)) * 2  # Two-tailed test
        
        # Append individual company results for this event date
        company_results.append({
            'company_symbol': company_symbol,
            'race_date': race_date,
            'abnormal_volume': company_abnormal_volume,
            'test_statistic': company_test_statistic,
            'p_value': company_p_value
        })
    
    
    # Calculate equal-weighted portfolio mean abnormal trading volume on the event date
    portfolio_mean_volume = event_window[abnormal_volume_cols].mean(axis=1).loc[event_date]
    portfolio_test_statistic = portfolio_mean_volume / std_dev
    portfolio_p_value = (1 - t.cdf(abs(portfolio_test_statistic), df=degrees_of_freedom)) * 2  # Two-tailed test
    
    # Append portfolio results for this event date
    portfolio_results.append({
        'race_date': race_date,
        'portfolio_mean_volume': portfolio_mean_volume,
        'test_statistic': portfolio_test_statistic,
        'p_value': portfolio_p_value
    })
    

# Convert results to DataFrame
portfolio_results_df = pd.DataFrame(portfolio_results)
company_results_df = pd.DataFrame(company_results)

# Display results
print(portfolio_results_df)
print(company_results_df)