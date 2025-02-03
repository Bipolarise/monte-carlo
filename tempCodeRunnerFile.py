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
from scipy.optimize import differential_evolution
from scipy.stats import truncnorm


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

def simulate_heston_model(S_0, mu, kappa, theta, sigma_v, v0, T, n_steps, rho, n_paths):
    """
    Simulates the Heston volatility model with correlation.
    
    Parameters:
    - S_0: float
        Initial stock price
    - mu: float
        Drift rate of the stock price
    - kappa: float
        Mean reversion rate of variance
    - theta: float
        Long-term mean variance
    - sigma_v: float
        Volatility of variance
    - v0: float
        Initial variance
    - T: float
        Time to maturity (in years)
    - N: int
        Number of time steps
    - rho: float
        Correlation between the Brownian motions for stock price and variance
    
    Returns:
    - v: pd.Series
        The volatility path of the stock
    - S: pd.Series
        The simulated stock price path
    """
    if rho > 1 or rho < -1:
        raise ValueError("Correlation must be between -1 and 1")
    
    dt = T / n_steps

    # Initialize arrays for paths
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    # Set initial conditions
    S[:, 0] = S_0
    v[:, 0] = np.maximum(v0, 1e-8)

    for i in range(1, n_steps + 1):
        dW1 = np.random.randn(n_paths) * np.sqrt(dt)
        dZ = np.random.randn(n_paths) * np.sqrt(dt)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ

        v[:, i] = np.maximum(v[:, i-1] + kappa * (theta - v[:, i-1]) * dt + sigma_v * np.sqrt(v[:, i-1]) * dW2, 1e-8)
        S[:, i] = S[:, i-1] * (1 + mu * dt + np.sqrt(v[:, i-1]) * dW1)
        
    plt.figure(figsize = (12,7))
    plt.plot(S)
    plt.show()

    return v, S

v, S = simulate_heston_model(S_0=100, mu=0.05, kappa=1.5, theta=0.04, sigma_v=0.3, v0=0.04, T=1, n_steps=252, rho=-0.7, n_paths=1000)