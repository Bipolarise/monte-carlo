import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# Step 1: Download data and calculate log returns
def prepare_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)['Adj Close']
    data = pd.DataFrame(data)
    data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    data['Squared Log Returns'] = data['Log Returns']**2
    return data.dropna()

data = prepare_data(ticker = 'PLTR', start = '2023-01-01', end = '2024-01-01')
log_returns = data['Log Returns']
initial_guess = [data['Log Returns'].mean(), data['Log Returns'].std()]



def neg_log_likelihood(params, returns):
    mu, sigma = params[0], params[1]
    n = len(returns)
    log_likelihood = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * np.log(sigma**2)
        - np.sum((returns - mu)**2) / (2 * sigma**2)
    )
    return -log_likelihood


result = minimize(
    neg_log_likelihood,
    initial_guess,
    args=(log_returns,),
    bounds=[(-np.inf, np.inf), (1e-6, np.inf)]  # Ensure sigma > 0
)

mu_mle, sigma_mle = result.x

mu_annualized = mu_mle * 252

print(f"Estimated mu (daily): {mu_mle}")
print(f"Estimated mu (annualized): {mu_annualized}")
print(f"Estimated sigma (daily): {sigma_mle}")
print(f"Estimated sigma (annualized): {sigma_mle * np.sqrt(252)}")

sample_mean = np.mean(log_returns)
print(sample_mean)
