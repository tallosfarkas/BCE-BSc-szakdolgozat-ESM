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
prediction_start_date = '2023-05-01'
prediction_end_date = '2023-09-09'

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
