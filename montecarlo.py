import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM

# Exemplar Testing Constants
gbm_params = {
    "mu": 0.05  # Drift
}

high_vol_params = {
    "v0": 0.04,
    "kappa": 2.0,
    "theta": 0.04,
    "sigma_v": 0.3
}

S0 = 100  # Starting stock price
T = 1.0  # 1 year
dt = 0.01  # Time step size
n_paths = 100  # Number of paths

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
                When converted to a Pandas DataFrame, each consecutive row represents
                
    NOTE: Need to incorporate rho, mu
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
    v0      :

    Returns:
    vol_paths : Simulated volatility paths (square root of variance).
    """
    n_steps = int(T/dt)
    stock_paths = np.zeros((n_paths, n_steps + 1))
    stock_paths[:,0] = S_0
    
    vol_paths = heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths)
    
    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_paths)  # Independent noise for each path
        stock_paths[:, i] = stock_paths[:, i - 1] * np.exp(
            (mu - 0.5 * vol_paths[:, i - 1] ** 2) * dt +
            vol_paths[:, i - 1] * Z * np.sqrt(dt)
        )

    return stock_paths, vol_paths


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

def calculate_log_returns(universe_data):
    for ticker in universe_data.columns:
        
        prices = universe_data[ticker]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
    return log_returns


def calculate_neg_log_likelihood(params, variance, dt = 1/252):
    """
    Finds the negative log likelihood of the Heston variance model.
    
    Parameters:
    params: list of Heston parameters, [kappa, theta, sigma_v]
    variance: observed realized variance
    dt: time step (daily)
    """
    kappa, theta, sigma_v = params
    
    v_t = variance.iloc[0]
    n = len(variance)
    log_likelihood = 0
    
    for t in range(1,n):
        
        mu = v_t + kappa * (theta - v_t) * dt
        var = sigma_v ** 2 * v_t * dt 
        observed = variance.iloc[t]
    
        log_likelihood += -0.5 * np.log(2 * math.pi * var) - 0.5 * ((observed - mu) ** 2 / var)
        
        v_t = observed
        v_t = max(v_t, 1e-6)

        var = (sigma_v ** 2 * v_t * dt) + 1e-6 

        
    return -log_likelihood



def Black_Scholes_Compare(S_t, K, r, t, sigma):
    """
    Finds the option price according to the Black Scholes Formula
    
    Parameters:
    S_t     : Pandas DataFrame of current prices for each stock
    K       : Pandas DataFrame of current strike prices for each stock
    r       : Current risk free rate
    t       : Time to expiration (years)
    sigma   : Current volatility of underlying asset
    """
    
    d1 = (np.log(S_t/K) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    C = norm.cdf(d1) * S_t - norm.cdf(d2) * K * np.exp(-r * t)
    
    return C


def use_BSM_Compare():
    if __name__ == "__main__":
        
        tickers = ['AAPL', 'MSFT', 'GOOGL']  

        stock_data = yf.download(tickers, start="2023-01-01", end="2023-12-31")
        adj_close_prices = stock_data['Adj Close'].iloc[-1]

        # Define test data
        strike_prices = pd.Series([240, 300, 150], index=tickers)  # Example strike prices
        risk_free_rate = 0.01
        time_to_expiration = 1.0
        volatility = 0.2

       
        option_prices = Black_Scholes_Compare(
            S_t=adj_close_prices,
            K=strike_prices,
            r=risk_free_rate,
            t=time_to_expiration,
            sigma=volatility
        )

        
        print("Option Prices:")
        print(option_prices)
        
# THIS IS A TEST FUNCTION (NOT COMPLETE)
def monte_carlo_simulation():
    
    # Example usage:
    S_0 = 130.69     # Initial stock price
    mu = 0.05        # Drift
    T = 1.0          # Time horizon (1 year)
    dt = 0.01        # Time step
    n_paths = 100    # Number of simulation paths

    # Heston model parameters
    v0 = 0.04        # Initial variance (vol^2)
    kappa = 2.0      # Mean-reversion rate
    theta = 0.04     # Long-term mean variance
    sigma_v = 0.3    # Volatility of variance (vol of vol)

    # Simulate GBM with Heston volatility
    S_T, vol_paths = Geometric_Brownian_Motion(S_0, mu, T, dt, n_paths, v0, kappa, theta, sigma_v)

    vol_paths = pd.DataFrame(vol_paths)
    # Print results
    for i in range(n_paths):
        print(f"Path {i+1} - Final Stock Price: {S_T[i, -1]:.2f}")
        
    K = 100
    r = 0.05
    payoffs = np.maximum(S_T - K, 0)
    discounted_payoffs = np.exp(-r * T) * payoffs

    # Monte Carlo option price
    option_price = np.mean(discounted_payoffs)
    print(f"Option Price: {option_price}")
    print(f"Volatility Path: {vol_paths}")




def simulate_HMM_Regime(start = '2000-01-01', end = '2023-01-01'):
    """
    Simulates the Hidden Markov Model for Market Regime Prediction. The model will output a binary variable for each day, 
    indicating whether it is high or low volatility. It will be trained with S&P 500 data. 
    
    Parameters:
    Start: A string containing the starting date
    End:  A string containing the ending date
    
    Returns:
    Vol_Day (Pandas Series): A DateTime series that contains a binary variable for each day
    """
    SP500 = yf.download('^GSPC', start = start, end = end)['Adj Close']

    SP500_returns = np.log(SP500 / SP500.shift(1)).dropna()
    X = SP500_returns.values.reshape(-1, 1)
    
    hmm = GaussianHMM(n_components = 2, covariance_type="diag", n_iter = 1000)
    hmm.fit(X)
    
    historical_regimes = hmm.predict(X)
    historical_regimes = pd.DataFrame(historical_regimes)
    historical_regimes = historical_regimes.set_index(SP500_returns.index)
    
    # TEST THIS FUNCTION
    return hmm, historical_regimes, SP500_returns, X
    
hmm, historical_regimes, SP500_returns, X = simulate_HMM_Regime(start = '2000-01-01', end = '2023-01-01')


def simulate_future_regimes(hmm, n_steps = 100, last_date = '2023-01-01'):
    """
    Simulates/Predicts future market regimes using a trained Hidden Markov Model from probabilities trained from
    simulate_HMM_Regime
    
    Parameters:
    hmm: Trained Gaussian HMM model
    n_steps: Number of future time steps to predict
    
    Returns:
    future_regimes: A Pandas Series for the sequence of future time steps
    """
    
    transition_matrix = hmm.transmat_
    current_state = hmm.predict(X)[-1]
    
    future_regimes = []
    for day in range(n_steps):
        next_state = np.random.choice([0,1], p = transition_matrix[current_state])
        future_regimes.append(next_state)
        current_state = next_state
        
    future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=n_steps)
    future_regimes = pd.Series(future_regimes, index=future_dates, name='Regime')
    
    return future_regimes


future_regimes = simulate_future_regimes(hmm, n_steps = 100)

# NEED TO IDENTIFY WHETHER 0 OR 1 IS HIGH VOL

def identify_hl_vol(historical_regimes, SP500_returns):
    """
    Identifies whether 0 or 1 is high or low volatility. Converts the existing future_regimes Series into either "High Volatility"
    or "Low Volatility"
    
    Parameters:
    future_regimes: Output from the simulate_future_regimes function
    SP500_returns: The Pandas Series for the SP500 returns, as obtained from the simulate_HMM_regime function
    
    Returns:
    dict: A dictionary for whether 0 or 1 is High or Low Volatility Regime
    
    NOTE: Prediction Series length must align with the SP500_returns length
    """
    historical_regimes.index = SP500_returns.index
    
    group_0 = SP500_returns[historical_regimes[0]  == 0]
    group_1 = SP500_returns[historical_regimes[0]  == 1]
    
    print("Group 0:")
    print(group_0)
    
    print("\n Group 1:")
    print(group_1)
    
    g1_returns = SP500_returns.loc[group_1.index]
    g0_returns = SP500_returns.loc[group_0.index]
    
    if g0_returns.std() > g1_returns.std():
        result = {
            "High Volatility": g0_returns,
            "Low Volatility": g1_returns,
            "High Vol Regime": 0,
            "Low Vol Regime": 1
        }
    else:
        result = {
            "High Volatility": g1_returns,
            "Low Volatility": g0_returns,
            "High Vol Regime": 1,
            "Low Vol Regime": 0
        }
        
    high_vol_regime = result["High Vol Regime"]
    low_vol_regime = result["Low Vol Regime"]
    
    future_volatility = future_regimes.replace({
        high_vol_regime: "High Volatility",
        low_vol_regime: "Low Volatility"
    })
        
    return future_volatility


def simulate_stock_prices(S0, T, dt, n_paths, high_vol_params, low_vol_params):
    """
    Combines the Hidden Markov Model (HMM) with the Heston volatility model, when volatility is high, and the Geometric Brownian Motion
    Model, when volatility is low.
    
    Parameters:
    
    """
    future_volatility = identify_hl_vol(historical_regimes, SP500_returns)
    
    n_steps = int(T/dt)
    simulated_prices = np.zeros((n_paths, n_steps + 1))
    simulated_prices[:, 0] = S0
    
    for t, regime in enumerate(future_volatility):
        print(f"Step {t}, Regime: {regime}")

        if regime == "High Volatility":
            
            # Heston Model Parameters
            v0 = high_vol_params["v0"]
            kappa = high_vol_params["kappa"]
            theta = high_vol_params["theta"]
            sigma_v = high_vol_params["sigma_v"]

            # Simulate using the Heston model for this step
            vol_paths = heston_volatility_process(v0, kappa, theta, sigma_v, T, dt, n_paths)
            for i in range(1, n_steps + 1):
                Z = np.random.normal(size=n_paths)
                simulated_prices[:, i] = simulated_prices[:, i - 1] * np.exp(
                    (-0.5 * vol_paths[:, i - 1] ** 2) * dt +
                    vol_paths[:, i - 1] * Z * np.sqrt(dt)
                )
        
        elif regime == "Low Volatility":
            mu = low_vol_params['mu']
            
            stock_paths, _ = Geometric_Brownian_Motion(S0, mu, T, dt, n_paths,
                                                       v0 = high_vol_params["v0"],
                                                       theta=high_vol_params["theta"],
                                                       sigma_v=high_vol_params["sigma_v"],
                                                       kappa = high_vol_params["kappa"])
            
            simulated_prices[:, t+1] = stock_paths[:, t+1]
       
            
    return pd.DataFrame(simulated_prices)
    
simulated_prices = simulate_stock_prices(S0, T, dt, n_paths, high_vol_params, gbm_params)
print(simulated_prices)

plt.figure(figsize=(12, 7))
for i in range(simulated_prices.shape[0]):
    plt.plot(simulated_prices.iloc[i, :], alpha=0.7)
plt.title("100 Simulations of Stock Prices")
plt.ylabel("Price in $")
plt.xlabel("Time Steps")
plt.show()


def run_model():    
    # Download Real Data

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

    log_returns = calculate_log_returns(all_data)
    realized_variance = log_returns ** 2

    initial_guess = [2.0, realized_variance.mean(), 0.1]
    bounds = [(0.01, 5), (0.0001, 0.2), (0.01, 1)]

    result = minimize(calculate_neg_log_likelihood, initial_guess, args=(realized_variance,), bounds=bounds)
    kappa_est, theta_est, sigma_v_est = result.x
    print(f"Estimated Parameters:")
    print(f"  kappa: {kappa_est:.4f}")
    print(f"  theta: {theta_est * np.sqrt(252):.6f}")
    print(f"  sigma_v: {sigma_v_est:.6f}")
    

    # for ticker, mu in mu_values.items():
    #     print(f"{ticker}: mu = {mu:.6f}")





#123456
# cd monte-carlo
# git add .
# git commit -m "smthn"
# git push