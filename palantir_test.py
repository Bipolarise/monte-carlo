import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# Define the Heston simulation function
def simulate_heston_variance(kappa, theta, sigma_v, v0, dt, n_steps):
    np.random.seed(42)  # For reproducibility
    v = np.zeros(n_steps)
    v[0] = v0
    for t in range(1, n_steps):
        dW_v = np.random.normal(0, np.sqrt(dt))  # Wiener increment
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + sigma_v * np.sqrt(max(v[t-1], 0)) * dW_v
        v[t] = max(v[t], 0)  # Ensure non-negative variance
    return v

# Define the log-likelihood function
def log_likelihood_heston(params, observed_v, dt):
    kappa, theta, sigma_v, v0 = params
    n_steps = len(observed_v)
    simulated_v = simulate_heston_variance(kappa, theta, sigma_v, v0, dt, n_steps)
    ll = 0
    for t in range(1, n_steps):
        mu = simulated_v[t-1] + kappa * (theta - simulated_v[t-1]) * dt
        variance = sigma_v**2 * simulated_v[t-1] * dt
        variance = max(variance, 1e-10)  # Prevent division by zero or log(0)
        ll += -0.5 * (np.log(2 * np.pi * variance) + (observed_v[t] - mu)**2 / variance)
    return -ll

# Prepare stock data
def prepare_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)['Adj Close']
    data = pd.DataFrame(data)
    data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Variance'] = data['Log Returns'].rolling(window=15).std()**2  # Rolling variance
    return data.dropna()

# Function to estimate parameters for a single stock
def estimate_heston_parameters(ticker, start='2023-01-01', end='2024-01-01'):
    try:
        data = prepare_data(ticker, start, end)
        observed_v = data['Variance'].values
        dt = 1 / 252  # Daily time steps
        initial_guess = [1.0, observed_v.mean(), observed_v.std(), observed_v[0]]  # [kappa, theta, sigma_v, v0]
        bounds = [(0.01, 5), (0.001, 1), (0.01, 5), (0.001, 1)]  # Parameter bounds
        result = minimize(log_likelihood_heston, initial_guess, args=(observed_v, dt), bounds=bounds)
        return result.x  # Return the estimated parameters
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return [None, None, None, None]

# List of stocks in the universe
universe = ['AAPL', 'MSFT', 'GOOG', 'KO']  # Replace with your stock tickers

# Compile results into a DataFrame
results = []
for ticker in universe:
    params = estimate_heston_parameters(ticker)
    results.append({
        'Ticker': ticker,
        'Kappa': params[0],
        'Theta': params[1],
        'Sigma_v': params[2],
        'V0': params[3]
    })

# Convert to Pandas DataFrame
results_df = pd.DataFrame(results)
print(results_df)


