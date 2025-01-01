import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths):
    n_steps = int(T / dt)
    vol_paths = np.zeros((n_paths, n_steps + 1))
    var_paths = np.zeros((n_paths, n_steps + 1))
    
    var_paths[:, 0] = v0
    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_paths)
        var_paths[:, i] = var_paths[:, i - 1] + kappa * (theta - var_paths[:, i - 1]) * dt + sigma_v * np.sqrt(var_paths[:, i - 1]) * np.sqrt(dt) * Z
        var_paths[:, i] = np.maximum(var_paths[:, i], 0)  # Ensure non-negative variance
        vol_paths[:, i] = np.sqrt(var_paths[:, i])
    
    return vol_paths

def geometric_brownian_motion(S_0, mu, T, dt, n_paths, vol_paths):
    n_steps = int(T / dt)
    stock_paths = np.zeros((n_paths, n_steps + 1))
    stock_paths[:, 0] = S_0

    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_paths)
        stock_paths[:, i] = stock_paths[:, i - 1] * np.exp(
            (mu - 0.5 * vol_paths[:, i - 1]**2) * dt +
            vol_paths[:, i - 1] * Z * np.sqrt(dt)
        )
    return stock_paths

# Parameters
S_0 = 150  # Initial stock price for AAPL
mu = 0.08  # Positive drift for upward trend
T = 1.0  # Time horizon (1 year)
dt = 1 / 252  # Daily time step
n_paths = 100  # Number of simulation paths
v0 = 0.04  # Initial variance (20% annualized volatility)
kappa = 2.0  # Mean reversion speed
theta = 0.04  # Long-term variance (20% annualized volatility)
sigma_v = 0.3  # Volatility of volatility

# Simulate volatility paths
vol_paths = heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths)

# Simulate stock prices
stock_paths = geometric_brownian_motion(S_0, mu, T, dt, n_paths, vol_paths)

# Plot results
plt.figure(figsize=(12, 6))
for i in range(n_paths):
    plt.plot(stock_paths[i, :], alpha=0.7, linewidth=0.7)
plt.title("Simulated Stock Prices for AAPL with Trend")
plt.xlabel("Time Steps")
plt.ylabel("Price ($)")
plt.show()
