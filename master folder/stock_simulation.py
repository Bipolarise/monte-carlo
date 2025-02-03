import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def simulate_heston_model(S_0, mu, kappa, theta, sigma_v, v0, T, N, rho):
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
    dt = T / N
    dW1 = np.random.randn(N) * np.sqrt(dt)
    dZ = np.random.randn(N) * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ

    S = np.zeros(N + 1)
    v = np.zeros(N + 1)
    S[0] = S_0
    v[0] = max(v0, 1e-8)

    for i in range(1, N + 1):
        v[i] = max(v[i-1] + kappa * (theta - v[i-1]) * dt + sigma_v * np.sqrt(v[i-1]) * dW2[i-1], 1e-8)
        S[i] = S[i-1] * (1 + mu * dt + np.sqrt(v[i-1]) * dW1[i-1])

    return v, S



def simulate_stock_prices(model_parameters, future_volatility, n_paths, dt, T):
    """
    Parameters:
    - model_params: pd DataFrame
        All parameter models, outputted from the all_params function
    - T: 
        Time to expiration (in years)
    - N: 
        Number of simulations
    
    Returns:
    - simulated_prices: pd DataFrame
    
    """
    n_steps = int(T / dt)
    if len(future_volatility) != n_steps:
        raise ValueError("Length of future_volatility does not match n_steps.")

    simulated_prices = {}

    for _, row in model_parameters.iterrows():
        stock = row['Stock']
        kappa = row['Kappa']
        theta = row['Theta']
        sigma_v = row['Sigma_v']
        v0 = row['v0']
        heston_mu = row['Heston Mu']
        starting_price = row['Initial Price']
        rho = row['Rho']

        # Generate independent vol_paths for each path
        vol_paths = np.zeros((n_paths, n_steps + 1))
        for i in range(n_paths):
            vol_paths[i, :] = simulate_heston_model(
                starting_price, heston_mu, kappa, theta, sigma_v, v0, T, n_steps, rho
            )[0]

        stock_paths = np.zeros((n_paths, n_steps + 1))
        stock_paths[:, 0] = starting_price

        for t in range(1, n_steps + 1):
            regime = future_volatility.iloc[t-1]
            Z = truncnorm.rvs(-3, 3, size=n_paths)  # Truncated normal to avoid extreme outliers

            if regime == "High Volatility":
                stock_paths[:, t] = stock_paths[:, t-1] * np.exp(
                    (-0.5 * vol_paths[:, t-1]**2) * dt +
                    vol_paths[:, t-1] * Z * np.sqrt(dt)
                )
            else:
                stock_paths[:, t] = stock_paths[:, t-1] * np.exp(
                    (heston_mu - 0.5 * vol_paths[:, t-1]) * dt +
                    np.sqrt(vol_paths[:, t-1]) * Z * np.sqrt(dt)
                )
            
        

    simulated_prices[stock] = stock_paths
    plt.figure(figsize=(12,7))
    plt.plot(simulated_prices[stock])
    plt.show()

    return simulate_stock_prices
