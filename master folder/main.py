import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm
import pandas_datareader.data as pdr
from datetime import datetime

from stock_simulation import *
from parameters import all_params
print(all_params)
from regimes import *


def download_prepare_data(universe, start, end):
    """
    Downloads the stocks in the universe and formats them for use.
    
    Parameters:
    - universe: list
        A list of stock tickers to be analyzed.
    - start: str
        The start date.
    - end: str
        The end date.
        
    Returns:
    - all_data: pd.DataFrame
        A Pandas DataFrame containing adjusted close prices and log returns for all stocks in the universe.
    """
    all_data = pd.DataFrame()
    
    for ticker in universe:
        data = yf.download(ticker, start=start, end=end)['Adj Close']
        data = pd.DataFrame(data)
        data.columns = [f'{ticker}_Adj Close']  
        
       
        data[f'{ticker}_Log Returns'] = np.log(data[f'{ticker}_Adj Close'] / data[f'{ticker}_Adj Close'].shift(1))
        
        
        all_data = pd.concat([all_data, data], axis=1)
    
    return all_data.dropna()

def run_model():
    
    global T, dt, N, threshold_factor, n_paths
    
    universe = ['AAPL', 'GOOG', 'NVDA']
    start_date, end_date = '2020-01-01', '2024-12-31'  # FOR TRAINING
    
    all_training_data = download_prepare_data(universe=universe, start=start_date, end=end_date)
    
    
    T = 1.0                 # Time to maturity in years
    dt = 1/252              # DO NOT CHANGE
    N = int(T/dt)           # DO NOT CHANGE
    threshold_factor = 1
    n_paths = 100
    

    model_parameters = all_params(price_data = all_training_data, dt = dt, threshold_factor = threshold_factor)
    future_volatility = hidden_markov_model(N, end_date)
    
    for index, row in model_parameters.iterrows():
    
        stock = row['Stock']  # Access the stock ticker
        kappa = row['Kappa']
        theta = row['Theta']
        sigma_v = row['Sigma_v']
        v0 = row['v0']
        gbm_mu = row['GBM Mu']
        starting_price = row['Initial Price']  # Ensure this is a scalar value
        poisson_mu = row['Mu Daily']
        poisson_sigma = row['Sigma Daily']
        poisson_lambda = row['Lambda Daily']
    
    X = simulate_stock_prices(model_parameters, future_volatility, n_paths, dt = dt, T = T)

    
run_model()    