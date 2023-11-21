import eikon as ek
import numpy as np
import arch
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error




############################ event study ################################################


############################# F1 data ######################################

import requests

# Defining a function to get the winner of a race in a particular year
def get_race_winner(year, race_number):
    url = f"http://ergast.com/api/f1/{year}/{race_number}/results.json"
    response = requests.get(url)
    data = response.json()
    
    # Extracting the constructor name of the race winner
    try:
        winner_constructor = data['MRData']['RaceTable']['Races'][0]['Results'][0]['Constructor']['name']
        return winner_constructor
    except (IndexError, KeyError):
        return None

# Defining a function to get all winners in a year
def get_all_winners_in_year(year):
    winners = []
    race_number = 1
    
    while True:
        winner = get_race_winner(year, race_number)
        if winner is None:
            break
        winners.append(winner)
        race_number += 1
    
    return winners

# Fetching the winners for the year 2014 as a starting point
winners_2014 = get_all_winners_in_year(2014)
winners_2014



import matplotlib.pyplot as plt

# Aggregate winners from all years
all_winners = []
for year in range(2014, 2022):
    all_winners.extend(get_all_winners_in_year(year))

# Count the number of wins by each team
win_counts = {}
for winner in all_winners:
    win_counts[winner] = win_counts.get(winner, 0) + 1

# Create a bar chart
teams = list(win_counts.keys())
wins = list(win_counts.values())

plt.bar(teams, wins)
plt.xlabel('Team')
plt.ylabel('Number of Wins')
plt.title('Number of Wins by Team (2014-2021)')
plt.xticks(rotation=45)
plt.show()








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

all_race_dates_and_names = get_all_race_dates_and_names(2014, 2022)
all_race_dates = [date for date, _ in all_race_dates_and_names]
all_race_dates = pd.to_datetime(all_race_dates)



import requests
import matplotlib.pyplot as plt
import numpy as np

# ... previous functions here ...

# Initialize a dictionary to hold win counts for each year
win_counts_by_year = {year: {} for year in range(2014, 2022)}

# Populate the win_counts_by_year dictionary
for year in range(2014, 2022):
    winners = get_all_winners_in_year(year)
    for winner in winners:
        win_counts_by_year[year][winner] = win_counts_by_year[year].get(winner, 0) + 1

# Get a list of all teams
all_teams = set()
for year, win_counts in win_counts_by_year.items():
    all_teams.update(win_counts.keys())

all_teams = sorted(list(all_teams))

# Convert win count data to a format suitable for plotting
data = []
for team in all_teams:
    wins_by_team = []
    for year in range(2014, 2022):
        wins_by_team.append(win_counts_by_year[year].get(team, 0))
    data.append(wins_by_team)

data = np.array(data)

# Create a stacked bar chart
years = range(2014, 2022)
bottom = np.zeros(len(years))
for i, team in enumerate(all_teams):
    plt.bar(years, data[i], bottom=bottom, label=team)
    bottom += data[i]

plt.xlabel('Year')
plt.ylabel('Number of Wins')
plt.title('Number of Wins by Team (2014-2021)')
plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(years)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

team_colors = {
    'Mercedes': '#00D2BE',
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'AlphaTauri': '#20394C',
    'Alpine F1 Team': '#2173B8',
    'McLaren': '#FF8000',
    'Racing Point': '#EC0374',
    'Red Bull':'#FFCC00',
    
    
    
    
}

data_df = pd.DataFrame(data, index=all_teams, columns=years)

# Sort the columns
data_df = data_df.sort_values(by=list(data_df.columns), axis=0, ascending=False)

all_teams = data_df.index.tolist()


fig, ax = plt.subplots(figsize=(10, 6))
data_df.T.plot(kind='bar', stacked=True, color=[team_colors[team] for team in all_teams], ax=ax)
plt.title('Number of Wins per Team per Year (2014-2021)')
plt.xlabel('Year')
plt.ylabel('Number of Wins')
plt.legend(title='Team')
plt.show()




import requests
import pandas as pd
import matplotlib.pyplot as plt

# Function to get constructor standings for a given year
def get_constructor_standings(year):
    url = f"http://ergast.com/api/f1/{year}/constructorStandings.json"
    response = requests.get(url)
    data = response.json()
    standings = data['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
    return [(item['Constructor']['constructorId'], float(item['points'])) for item in standings]

# Get constructor standings for each year from 2014 to 2021
all_standings = {year: get_constructor_standings(year) for year in range(2014, 2022)}

# Convert to DataFrame for easier manipulation
df_list = []
for year, standings in all_standings.items():
    df = pd.DataFrame(standings, columns=['Team', 'Points'])
    df['Year'] = year
    df_list.append(df)

# Concatenate all dataframes
all_data = pd.concat(df_list)

# Pivot to get a matrix of team points per year
pivot_data = all_data.pivot(index='Year', columns='Team', values='Points').fillna(0)


import random

# Melt the DataFrame to long format
long_data = pivot_data.reset_index().melt(id_vars='Year', var_name='Team', value_name='Points')

# Sort by Year and Points (in descending order within each year)
long_data = long_data.sort_values(by=['Year', 'Points'], ascending=[True, False])

# Pivot back to wide format
sorted_pivot_data = long_data.pivot(index='Year', columns='Team', values='Points')

# Generate random color list
color_list = ["#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(len(sorted_pivot_data.columns))]

# Now proceed with plotting
sorted_pivot_data.plot(kind='bar', stacked=True, figsize=(10,6), color=color_list)
plt.title('Number of Points per Team per Year (2014-2021)')
plt.xlabel('Year')
plt.ylabel('Number of Points')
plt.legend(title='Team')
plt.show()







############################## AR and AAR ##############################


trading_days = abnormal_returns_mercedes.index

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
    aar = np.mean(ars)
    variance = np.var(ars)
    return aar, variance


def t_test_for_aar(aar, variance, n):
    t_stat = aar / (np.sqrt(variance / n))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    return t_stat, p_val

# Assuming abnormal_returns_mercedes is a DataFrame or Series with dates as the index
trading_days = abnormal_returns_mercedes.index

daily_windows = list(range(-5, 6))  # This will give [-5, -4, ... 0, ... 4, 5]
results_daily = []

for day_offset in daily_windows:
    aar, variance = calculate_aar_for_day(abnormal_returns_mercedes, all_race_dates, trading_days, day_offset)
    t_stat, p_val = t_test_for_aar(aar, variance, len(all_race_dates))
    results_daily.append({
        't': day_offset,
        'AAR': aar,
        't-ratio': t_stat,
        'p-value': p_val
    })

results_daily_df = pd.DataFrame(results_daily)
print(results_daily_df)









def calculate_ar_for_window(abnormal_returns, event_day, start_offset, end_offset):
    # Same as previous function, but rename it for clarity
    start_day = event_day + pd.Timedelta(days=start_offset)
    end_day = event_day + pd.Timedelta(days=end_offset)
    ar = abnormal_returns.loc[start_day:end_day]  # No sum, as we want daily AR
    return ar

def calculate_aar_for_window(abnormal_returns, all_race_dates, trading_days, start_offset, end_offset):
    ars = []
    for race_date in all_race_dates:
        event_day = find_next_trading_day(race_date, trading_days)
        ar = calculate_ar_for_window(abnormal_returns, event_day, start_offset, end_offset)
        ars.extend(ar)  # Use extend instead of append to flatten the list
    aar = np.mean(ars)  # Compute the mean for AAR
    variance = np.var(ars)
    return aar, variance

def t_test_for_caar(caar, variance, n):
    t_stat = caar / (variance / np.sqrt(n))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))  # Two-tailed test
    return t_stat, p_val

# Assuming abnormal_returns_mercedes is a DataFrame or Series with dates as the index
trading_days = abnormal_returns_mercedes.index


extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]
results_extended = []


for window in extended_windows:
    start, end = window
    aar, variance = calculate_aar_for_window(abnormal_returns_mercedes, all_race_dates, trading_days, start, end)
    t_stat, p_val = t_test_for_caar(aar, variance, len(all_race_dates))  # Update function name if needed
    results_extended.append({
        'Window': window,
        'AAR': aar,  # Update key to 'AAR'
        'Variance': variance,
        't-statistic': t_stat,
        'p-value': p_val
    })

results_extended_df_AAR = pd.DataFrame(results_extended)
print(results_extended_df_AAR)

# Assuming abnormal_returns_mercedes is defined from your previous code

################################# CAR and CAAR ############################################
def find_next_trading_day(date, trading_days):
    # Find the next trading day after the given date
    return trading_days[trading_days > date].min()



def calculate_car_for_window(abnormal_returns, event_day, start_offset, end_offset):
    # Calculate the start and end of the event window relative to the event_day
    start_day = event_day + pd.Timedelta(days=start_offset)
    end_day = event_day + pd.Timedelta(days=end_offset)
    
    # Sum the abnormal returns over the specific event window
    car = abnormal_returns.loc[start_day:end_day].sum()
    return car

def calculate_caar_for_window(abnormal_returns, all_race_dates, trading_days, start_offset, end_offset):
    cars = []
    for race_date in all_race_dates:
        # Find the next trading day after the race
        event_day = find_next_trading_day(race_date, trading_days)
        car = calculate_car_for_window(abnormal_returns, event_day, start_offset, end_offset)
        cars.append(car)
    caar = np.mean(cars)
    variance = np.var(cars)
    return caar, variance


def t_test_for_caar(caar, variance, n):
    t_stat = caar / (variance / np.sqrt(n))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))  # Two-tailed test
    return t_stat, p_val

# Assuming abnormal_returns_mercedes is a DataFrame or Series with dates as the index
trading_days = abnormal_returns_mercedes.index


extended_windows = [(-5,-1), (-1,1), (0,0), (0,1), (1,5)]
results_extended = []

for window in extended_windows:
    start, end = window
    caar, variance = calculate_caar_for_window(abnormal_returns_mercedes, all_race_dates, trading_days, start, end)
    t_stat, p_val = t_test_for_caar(caar, variance, len(all_race_dates))
    results_extended.append({
        'Window': window,
        'CAAR': caar,
        'Variance': variance,
        't-statistic': t_stat,
        'p-value': p_val
    })

results_extended_df_CAAR = pd.DataFrame(results_extended)
print(results_extended_df_CAAR)


###################### plotolás #################################################################################


#results_extended_df = results_extended_df_CAAR
results_extended_df = results_extended_df_AAR



import matplotlib.pyplot as plt

# Assuming results_extended_df is your DataFrame with the results
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plotting CAAR
axs[0].bar(results_extended_df['Window'].astype(str), results_extended_df['AAR'])
axs[0].set_title('CAAR across different event windows')
axs[0].set_ylabel('CAAR')

# Plotting t-statistic
axs[1].bar(results_extended_df['Window'].astype(str), results_extended_df['t-statistic'])
axs[1].set_title('t-statistic across different event windows')
axs[1].set_ylabel('t-statistic')

# Plotting p-value
axs[2].bar(results_extended_df['Window'].astype(str), results_extended_df['p-value'])
axs[2].set_title('p-value across different event windows')
axs[2].set_ylabel('p-value')
axs[2].set_xlabel('Event Windows')

plt.tight_layout()
plt.show()
