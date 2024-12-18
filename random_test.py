from scipy.optimize import minimize
import pandas as pd
import numpy as np
import yfinance as yf


from scipy.optimize import minimize
import numpy as np

def heston_mle_for_all_stocks(data, dt=1/252):
    """
    Estimate Heston parameters for all stocks in the dataset.

    Parameters:
    data: pd.DataFrame
        DataFrame where each column represents a stock's adjusted closing prices.
    dt: float
        Time step, default is 1 trading day (1/252).

    Returns:
    dict
        Dictionary of Heston model parameters for each stock.
    """
    def heston_mle_equations(params, variance):
        kappa, theta, sigma_v = params
        v_t = variance.iloc[0]  # Initial variance
        n = len(variance)
        neg_log_likelihood = 0

        for t in range(1, n):
            # Conditional mean and variance
            mu_t = v_t + kappa * (theta - v_t) * dt
            var_t = max(sigma_v**2 * v_t * dt, 1e-6)  # Add small epsilon for numerical stability
            
            # Observed variance
            observed_v = variance.iloc[t]

            # Negative log-likelihood for this step
            neg_log_likelihood += (
                0.5 * np.log(2 * np.pi * var_t) + 0.5 * ((observed_v - mu_t)**2 / var_t)
            )

            # Update v_t for the next step
            v_t = observed_v

        return neg_log_likelihood

    # Calculate log returns
    log_returns = np.log(data / data.shift(1)).dropna()
    realized_variance = log_returns**2  # Realized variance

    # Dictionary to store results
    stock_parameters = {}

    # Process each stock separately
    for stock in data.columns:
        stock_variance = realized_variance[stock]

        # Initial guess and bounds
        initial_guess = [1.0, stock_variance.mean(), 0.2]
        bounds = [(0.01, 5), (0.0001, 0.5), (0.01, 1)]

        # Optimize for this stock
        result = minimize(
            heston_mle_equations,
            initial_guess,
            args=(stock_variance,),
            bounds=bounds
        )

        # Store parameters
        if result.success:
            stock_parameters[stock] = {
                'kappa': result.x[0],
                'theta': result.x[1],
                'sigma_v': result.x[2],
            }
        else:
            stock_parameters[stock] = {
                'error': result.message
            }

    return stock_parameters


universe = [
    'NVDA'
]

all_data = {}
start = '2023-12-8'
end = '2024-12-8'

for stock in universe:
    stock_data = yf.download(stock, start = start, end = end)['Adj Close']
    all_data[stock] = stock_data
    
all_data = pd.DataFrame(all_data)
    
# Call the function on your data
parameters_per_stock = heston_mle_for_all_stocks(all_data)

# Print results
for stock, params in parameters_per_stock.items():
    print(f"{stock}: {params}")



    
    


'''
    
    

all_data = pd.DataFrame(all_data)

log_returns = np.log(all_data / all_data.shift(1)).dropna()
realized_variance = log_returns**2

from scipy.optimize import minimize

initial_guess = [1.0, realized_variance.mean().mean(), 0.2]
bounds = [(0.01, 5), (0.0001, 0.5), (0.01, 1)]

result = minimize(
    heston_mle_equations,
    initial_guess,
    args=(realized_variance.mean(axis=1),),
    bounds=bounds
)

print("Estimated Parameters:")
print(f"kappa: {result.x[0]:.4f}, theta: {result.x[1]:.6f}, sigma_v: {result.x[2]:.6f}")'''

