import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Heston Volatility Model

def heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths):
    """
    Simulates the volatility process from the Heston model.

    Parameters:
    v0      : Initial variance (v_0).
    kappa   : Mean-reversion rate.
    theta   : Long-term mean variance.
    sigma_v : Volatility of variance (vol of vol).
    T       : Time horizon.
    dt      : Time step.
    n_paths : Number of simulation paths.

    Returns:
    vol_paths : Simulated volatility paths (square root of variance).
    """
    
    n_steps = int(T/dt)
    vol_paths = np.zeros((n_paths, n_steps + 1))
    var_paths = np.zeros((n_paths, n_steps + 1))
    
    var_paths[:,0] = v0

    for i in range(1, n_steps + 1):
        Z = np.random.normal(0,1, size = n_paths)
        var_paths[:,i] = var_paths[:,i-1] + kappa * (theta - var_paths[:,i-1]) * dt + sigma_v * np.sqrt(var_paths[:,i-1]) * np.sqrt(dt) * Z
        var_paths[:,i] = np.maximum(var_paths[:,i], 0)
        
        vol_paths[:,i] = np.sqrt(var_paths[:,i])
    
    return vol_paths

# Geometric Brownian Motion (Simulation of Underlying Asset Price)
def Geometric_Brownian_Motion(S_0, mu, T, dt, n_paths, v0, kappa, theta, sigma_v):
    
    """
    Simulates stock prices according to the Geometric Brownian Motion.
    
    Parameters:
    S_0     : Initial Stock Price
    mu      : Percentage Drift
    sigma   : Volatility.
    T       : Terminal time
    dt      : Time step.
    n       : Number of simulation paths.

    Returns:
    vol_paths : Simulated volatility paths (square root of variance).
    """
    n_steps = int(T/dt)
    stock_paths = np.zeros((n_paths, n_steps + 1))
    stock_paths[:,0] = S_0
    
    vol_paths = heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths)
    
    for i in range(1, n_steps + 1):
        
        Z = np.random.normal(0, 1, size = n_paths)
        dt_sqrt = np.sqrt(dt)
        
        time = 0

        while (time + dt <= T):
            stock_paths[:,i] = stock_paths[:,i-1] * np.exp(
                (mu - 0.5 * vol_paths[:,i-1]**2) * dt + vol_paths[:,i-1] * np.random.normal(0, dt_sqrt)
                )
                        
            time += dt
            
        if T - time > 0:
            stock_paths[:,i] = stock_paths[:,i-1] * np.exp(
                (mu - 0.5 * vol_paths[:,i-1]**2) * (T - time) + vol_paths[:,i-1] * np.random.normal(0, np.sqrt(T - time))
                )
            
        
    return stock_paths, vol_paths

'''
# Example usage:
S_0 = 100        # Initial stock price
mu = 0.05        # Drift
T = 1.0          # Time horizon (1 year)
dt = 0.01        # Time step
n_paths = 20      # Number of simulation paths

# Heston model parameters
v0 = 0.04        # Initial variance (vol^2)
kappa = 2.0      # Mean-reversion rate
theta = 0.04     # Long-term mean variance
sigma_v = 0.3    # Volatility of variance (vol of vol)

# Simulate GBM with Heston volatility
stock_paths, vol_paths = Geometric_Brownian_Motion(S_0, mu, T, dt, n_paths, v0, kappa, theta, sigma_v)

# Print results
for i in range(n_paths):
    print(f"Path {i+1} - Final Stock Price: {stock_paths[i, -1]:.2f}")
'''




def calculate_mu(universe_data):
    """
    Calculate drift (mu) for each stock in the universe.

    Parameters:
    universe_data: DataFrame
        DataFrame where each column represents a stock's adjusted closing prices.

    Returns:
    mu_values: dict
        Dictionary where keys are stock tickers and values are drift (mu) estimates.
    """
    
    mu_values = {}
    
    for ticker, prices in universe_data.items():
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu_log = log_returns.mean() * 252
        
        sigma = log_returns.std() * np.sqrt(252)
        mu = mu_log + 0.5 * sigma ** 2
        mu_values[ticker] = mu
        
    return mu_values


def calculate_initial_variance(universe_data):
    """
    Calculate initial variance (sigma^2) for each stock in the universe.
    
    Parameters:
    universe_data: DataFrame - Columns are tickers, rows are dates, and values are prices.
    
    Returns:
    Dictionary of variances for each stock.
    """
    initial_variances = {}
    
    for ticker in universe_data.columns:
        
        prices = universe_data[ticker]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        variance = log_returns.var()
        
        initial_variances[ticker] = variance
    
    return initial_variances


        

# Download Real Data

universe = [
    'AAPL', 'NVDA'
]

all_data = {}
start = '2023-12-8'
end = '2024-12-8'

for stock in universe:
    stock_data = yf.download(stock, start = start, end = end)['Adj Close']
    all_data[stock] = stock_data
    

all_data = pd.DataFrame(all_data)
mu_values = calculate_initial_variance(all_data)

for ticker, mu in mu_values.items():
    print(f"{ticker}: mu = {mu:.6f}")
