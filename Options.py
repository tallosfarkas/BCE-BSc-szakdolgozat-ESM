import eikon as ek
import numpy as np
import arch
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import requests
import os

image_directory = "C:/Users/fajka/OneDrive - Corvinus University of Budapest/7. SZAKDOGA/Eredmenyek" # Change this to your desired path
excel_directory = "C:/Users/fajka/OneDrive - Corvinus University of Budapest/7. SZAKDOGA/Eredmenyek"  # Change this to your desired path

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
    frames.append(df)

combined_df = pd.concat(frames, axis=1)
combined_df.dropna(inplace=True)


# Function to get race dates and names
def get_race_results(year, race_number):
    url = f"http://ergast.com/api/f1/{year}/{race_number}/results.json"
    response = requests.get(url)
    data = response.json()
    
    race_date = data['MRData']['RaceTable']['Races'][0]['date']
    race_name = data['MRData']['RaceTable']['Races'][0]['raceName']
    
    return pd.DataFrame({'Race Date': [race_date], 'Race Name': [race_name]})

def get_all_race_dates_and_names(start_year, end_year):
    races = []
    for year in range(start_year, end_year + 1):
        race_number = 1
        while True:
            try:
                race_results = get_race_results(year, race_number)
                race_date = race_results['Race Date'][0]
                race_name = race_results['Race Name'][0]
                races.append((race_date, race_name))
                race_number += 1
            except IndexError:
                break
    return races

all_race_dates_and_names = get_all_race_dates_and_names(2021, 2023)
race_dates = pd.DataFrame(all_race_dates_and_names, columns=['Race Date', 'Race Name'])

# Ensure race dates are in datetime format
race_dates['Race Date'] = pd.to_datetime(race_dates['Race Date'])

# List of unique symbols in the combined_df based on column names
symbols = list(set(column.split('_')[0] for column in combined_df.columns if '_' in column))

for symbol in symbols:
    # Extracting specific data for the current symbol
    date_column = f"{symbol}_Date"
    volume_column = f"{symbol}_Volume"
    openint_column = f"{symbol}_OpenInt"
    
    data = combined_df[[date_column, volume_column, openint_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column])
    
    race_dates['Closest Trading Day'] = race_dates['Race Date'].apply(
        lambda race_date: data[data[date_column] > race_date][date_column].min()
    )
    
    merged_data = pd.merge(data, race_dates, left_on=date_column, right_on='Closest Trading Day', how='left')
   # Convert to numeric and drop NaNs
    merged_data[volume_column] = pd.to_numeric(merged_data[volume_column], errors='coerce')
    merged_data[openint_column] = pd.to_numeric(merged_data[openint_column], errors='coerce')
    merged_data.dropna(subset=[volume_column, openint_column], inplace=True)
    
    merged_data['Race_Dummy'] = merged_data['Closest Trading Day'].notna()
    
    
    
    # Conduct t-tests
    t_test_volume = stats.ttest_ind(merged_data[merged_data['Race_Dummy']][volume_column],
                                    merged_data[~merged_data['Race_Dummy']][volume_column])

    t_test_open_interest = stats.ttest_ind(merged_data[merged_data['Race_Dummy']][openint_column],
                                           merged_data[~merged_data['Race_Dummy']][openint_column])

    print(f'For symbol {symbol}:')
    print(f'T-test results for Volume: {t_test_volume}')
    print(f'T-test results for Open Interest: {t_test_open_interest}')

    merged_data['Race_Dummy'] = merged_data['Race_Dummy'].astype(int)
    
    # Run regression models:
    X = merged_data['Race_Dummy']
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    model_volume = sm.OLS(merged_data[volume_column], X).fit()
    model_open_interest = sm.OLS(merged_data[openint_column], X).fit()

    print(model_volume.summary())
    print(model_open_interest.summary())
    print('-'*50)
    # Visualization
    plt.figure(figsize=(10,5))
    plt.plot(merged_data[date_column], merged_data[volume_column], label='Volume')
    plt.plot(merged_data[date_column], merged_data[openint_column], label='Open Interest')
    
    # Plotting race dates as vertical lines
    for race_date in race_dates['Race Date']:
        plt.axvline(x=race_date, color='red', linestyle='--', alpha=0.7)  # Using a red dashed line for race dates
    
    plt.title(f"Data for {symbol}")
    plt.legend()
    plt.show()
    # Save the figure as an image
    plt.savefig(f"{image_directory}/{symbol}.jpg")

    # Save the merged_data DataFrame to an Excel file
    merged_data.to_excel(f"{excel_directory}/{symbol}_data.xlsx")

    print('-'*50)
    
    
    

    
    

