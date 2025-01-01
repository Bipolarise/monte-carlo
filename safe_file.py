import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

universe = ['NVDA', 'AAPL', 'KO']
start = '2022-01-01' # For Training data
end = '2024-12-01' # For Training Data

# NOTE: Recommended Large Training Size (10+ years)

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
    v0      : Initial Variance

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

def calculate_log_returns(universe_data):
    for ticker in universe_data.columns:
        
        prices = universe_data[ticker]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
    return log_returns


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


def simulate_HMM_Regime(start = start, end = end):
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


def simulate_future_regimes(hmm, n_steps, last_date = '2023-01-01'):
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


future_regimes = simulate_future_regimes(hmm, n_steps = 200)

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
    S_0            : The initial stock price
    T              : The time frame in years
    dt             : The time increment
    high_vol_params: Parameters for Heston Volatility Model (obtained dynamically for each stock)
    low_vol_params : Parameters for GBM model (obtained dynamically for each stock)
    
    NOTE: Parameters are sourced dynamically from `all_params`.
    """
    future_volatility = identify_hl_vol(historical_regimes, SP500_returns)
    
    n_steps = int(T/dt)
    simulated_prices = np.zeros((n_paths, n_steps + 1))
    simulated_prices[:, 0] = S0
    
    for t, regime in enumerate(future_volatility):
        if regime == "High Volatility":
            # Use high volatility parameters dynamically for the specific stock
            kappa = high_vol_params['Kappa']
            theta = high_vol_params['Theta']
            sigma_v = high_vol_params['Sigma_v']
            v0 = high_vol_params['V0']

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
                                                        v0=high_vol_params["V0"],
                                                        theta=high_vol_params["Theta"],
                                                        sigma_v=high_vol_params["Sigma_v"],
                                                        kappa=high_vol_params["Kappa"])
            simulated_prices[:, t+1] = stock_paths[:, t+1]
    
    return pd.DataFrame(simulated_prices)

    
'''simulated_prices = simulate_stock_prices(S0, T, dt, n_paths, high_vol_params, gbm_params)
final_prices = simulated_prices.iloc[:,-1:]
final_prices.index = future_regimes.index
'''
def convergence_analysis(simulated_prices, future_regimes=None):
    """
    Performs convergence analysis on the performed Monte Carlo simulation to estimate the final stock price at the end of the period.
    
    Parameters:
    simulated_prices: pd.DataFrame
        The Pandas DataFrame of simulated prices (rows = simulations, columns = time steps).
    future_regimes: pd.Series, optional
        The Pandas Series of future regimes to align the index (optional).
    
    Returns:
    lower_bound: pd.Series
        The lower confidence bounds.
    upper_bound: pd.Series
        The upper confidence bounds.
    final_mean: float
        The final mean of the simulation
    """

    # Extract final prices
    simulated_prices = pd.DataFrame(simulated_prices)
    final_prices = simulated_prices.iloc[:, -1]  

    if future_regimes is not None:
        if len(final_prices) != len(future_regimes):
            raise ValueError("Length of future_regimes must match the number of simulations.")
        final_prices.index = future_regimes.index

    n_simulations = len(final_prices)

    cumulative_mean = final_prices.expanding().mean()
    std_dev = final_prices.expanding().std()

    num_samples = np.arange(1, n_simulations + 1)

    # Calculate standard error
    standard_error = std_dev / np.sqrt(num_samples)

    confidence_level = 0.95
    z = norm.ppf((1 + confidence_level) / 2)
    lower_bound = cumulative_mean - z * standard_error
    upper_bound = cumulative_mean + z * standard_error
    
    final_mean = cumulative_mean.iloc[-1]

    # Optional Plotting
    # plt.figure(figsize=(12, 7))
    # plt.plot(cumulative_mean, label="Cumulative Mean", color="blue")
    # plt.fill_between(range(len(cumulative_mean)), lower_bound, upper_bound, color="blue", alpha=0.2, label="95% Confidence Interval")
    # plt.title("Convergence Analysis of Simulated Prices")
    # plt.xlabel("Number of Simulations")
    # plt.ylabel("Final Price")
    # plt.legend()
    # plt.show()

    return lower_bound, upper_bound, final_mean


def calculate_option_price(final_mean):
    """
    Calculates the price of the option given the predicted price
    
    Parameters:
    final_mean: int
        The final mean of the Monte Carlo Simulation of Stock Prices
    """

    option_payoff = np.max(final_mean - S0, 0)
    option_price = option_payoff * np.exp(-r*T)
    
    return option_price

'''plt.figure(figsize=(12, 7))

for i in range(simulated_prices.shape[0]):
    plt.plot(simulated_prices.iloc[i, :], alpha=0.7)
plt.title("100 Simulations of Stock Prices")
plt.ylabel("Price in $")
plt.xlabel("Time Steps")
plt.show()'''

# THIS IS A TEST FUNCTION (NOT COMPLETE)
def monte_carlo_simulation(n_batches, future_regimes = None):
    """
    Runs n-batches of Monte-Carlo Convergence Analyses. 
    
    Parameters:
    n_batches: int
        The number of batches to be simulated. For more accurate results, it is recommended that n_batches > 40
    
    NOTE: Each batch is of 100 simulations, i.e n_batches = 40 will simulate 4000. Changing the time frame is different. Changing 
    the number of simulations is comparable to the number of 'lines' the graph will have if graphed.
    
    Returns:
    mean_option_price
    """
        
    
    all_final_prices = []
    option_prices = []
    
    for _ in range(0, n_batches + 1):
        _, _, final_mean = convergence_analysis(simulated_prices, future_regimes=None)
        all_final_prices.append(final_mean)
        
    final_prices_mean = np.mean(all_final_prices)
    final_prices_std = np.std(all_final_prices)
    
    for price in all_final_prices:
        option_price = calculate_option_price(price)
        option_prices.append(option_price)
        
    mean_option_price = np.mean(option_prices)
    
    final_prices_mean = pd.Series(all_final_prices)
    option_prices = pd.Series(option_prices)
    
    
    return mean_option_price

def print_information():
    S0 = 100  
    T = 2.0 
    dt = 0.01  
    n_paths = 100  
    r = 0.05 

# sig = monte_carlo_simulation(n_batches=50)
# print(sig)
    
def download_prepare_data(universe, start, end):
    """
    Downloads the stocks in the universe and formats them for use.
    
    Parameters:
    universe: list
        A list of stock tickers to be analyzed.
    start: str
        The start date.
    end: str
        The end date.
        
    Returns:
    data: pd.DataFrame
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


def negative_log_likelihood_gbm(params, returns):
    """
    Returns the negative log likelihood for Maximum Likelihood Estimation
    """
    mu, sigma = params[0], params[1]
    n = len(returns)
    
    log_likelihood = (
        -n / 2 * np.log(2 * np.pi)
        - n / 2 * np.log(sigma**2)
        - np.sum((returns - mu)**2) / (2 * sigma**2)
    )
    
    
    return -log_likelihood

def calculate_mu(universe, start, end):
    """
    Calculates the GBM parameter drift using maximum likelihood estimation
    
    Parameters:
    universe: list
        A list of the stocks to be analysed
    start: str
        A string for the start date
    end: str
        A string for the end date
        
    Returns:
    results_df: pd DataFrame
        A Pandas DataFrame containing the different mu's for each stock, along with its volatility
    """
    all_data = download_prepare_data(universe=universe, start=start, end=end)
    log_returns = all_data.filter(like='Log Returns')

    results = []

    for stock in log_returns.columns:
        stock_returns = log_returns[stock].dropna()  

        # Initial guess for mu and sigma
        initial_guess = [stock_returns.mean(), stock_returns.std()]

        # Perform MLE using negative log-likelihood
        result = minimize(
            negative_log_likelihood_gbm,
            initial_guess,
            args=(stock_returns,),
            bounds=[(-np.inf, np.inf), (1e-6, np.inf)] 
        )

        mu_mle, sigma_mle = result.x

        results.append({
            'Stock': stock.replace('_Log Returns', ''),  
            'mu_daily': mu_mle,
            'mu_annualized': mu_mle * 252,
            'sigma_daily': sigma_mle,
            'sigma_annualized': sigma_mle * np.sqrt(252) 
        })

    results_df = pd.DataFrame(results)
    
    return results_df

def simulate_heston_variance(kappa, theta, sigma_v, v0, dt, n_steps):
    np.random.seed(42)  # For reproducibility
    v = np.zeros(n_steps)
    v[0] = v0
    for t in range(1, n_steps):
        dW_v = np.random.normal(0, np.sqrt(dt))  # Wiener increment
        v[t] = v[t-1] + kappa * (theta - v[t-1]) * dt + sigma_v * np.sqrt(max(v[t-1], 0)) * dW_v
        v[t] = max(v[t], 0)  
    return v

def log_likelihood_heston(params, observed_v, dt):
    kappa, theta, sigma_v, v0 = params
    n_steps = len(observed_v)
    simulated_v = simulate_heston_variance(kappa, theta, sigma_v, v0, dt, n_steps)
    ll = 0
    for t in range(1, n_steps):
        mu = simulated_v[t-1] + kappa * (theta - simulated_v[t-1]) * dt
        variance = sigma_v**2 * simulated_v[t-1] * dt
        variance = max(variance, 1e-10)  
        ll += -0.5 * (np.log(2 * np.pi * variance) + (observed_v[t] - mu)**2 / variance)
    return -ll

def estimate_heston_parameters(ticker, start = start, end = end):
    try:
        
        all_data = download_prepare_data([ticker], start=start, end=end)
        log_returns = all_data.filter(like='Log Returns')

        
        variance = log_returns[f'{ticker}_Log Returns'].rolling(window=15).var()
        observed_v = variance.dropna().values 
        dt = 1 / 252  

        initial_guess = [1.0, observed_v.mean(), observed_v.std(), observed_v[0]]  # [kappa, theta, sigma_v, v0]
        bounds = [(0.01, 5), (0.001, 1), (0.01, 5), (0.001, 1)] 

        # Perform optimization
        result = minimize(log_likelihood_heston, initial_guess, args=(observed_v, dt), bounds=bounds)
        return result.x  
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return [None, None, None, None]



def all_params():
    """
    Combines the mu values for GBM and parameters for the Heston Volatility model
    
    Returns:
    results_df: Pandas DataFrame
        A Pandas DataFrame containing all parameters required for the GBM and Heston Volatility
    """
    
    all_data = download_prepare_data(universe = universe, start = start, end = end)
    adj_close = all_data.filter(like='Adj Close').tail(1)
    last_stock_price = adj_close.iloc[-1].values
    
    mu_values = calculate_mu(universe = universe, start = start, end = end)
    results = []
    for ticker in universe:
        params = estimate_heston_parameters(ticker)
        results.append({
            'Stock': ticker,
            'Kappa': params[0],
            'Theta': params[1],
            'Sigma_v': params[2],
            'V0': params[3]
        })
    results_df = pd.DataFrame(results)
    
    results_df['mu_daily'] = mu_values['mu_daily'].values
    results_df['Starting Price'] = last_stock_price
    
    return results_df


def run_model():    
    """
    Integrates all parameters from the `all_params` function into the Monte Carlo convergence models.
    """
    # Retrieve model parameters for all stocks
    model_parameters = all_params()

    for _, row in model_parameters.iterrows():
        stock = row['Stock']
        kappa = row['Kappa']
        theta = row['Theta']
        sigma_v = row['Sigma_v']
        v0 = row['V0']
        mu_daily = row['mu_daily']
        starting_price = row['Starting Price']

        print(f"Running model for {stock}:")
        print(f"  Kappa: {kappa:.4f}, Theta: {theta:.6f}, Sigma_v: {sigma_v:.6f}, V0: {v0:.6f}, Mu: {mu_daily:.6f}, Starting Price: {starting_price:.2f}")

        # Use parameters in the Monte Carlo model
        n_paths = 100  # Number of simulation paths
        T = 1.0  # 1 year
        dt = 1 / 252  # Daily time step

        # Simulate stock prices using the Heston model
        simulated_prices, simulated_vols = Geometric_Brownian_Motion(
            S_0=starting_price,
            mu=mu_daily,
            T=T,
            dt=dt,
            n_paths=n_paths,
            v0=v0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v
        )

        # Perform convergence analysis
        lower_bound, upper_bound, final_mean = convergence_analysis(pd.DataFrame(simulated_prices))
        print(f"  Final Mean: {final_mean:.2f}")
        print(f"  95% Confidence Interval: ({lower_bound.iloc[-1]:.2f}, {upper_bound.iloc[-1]:.2f})")

        # Optional plotting
        plt.figure(figsize=(10, 6))
        plt.plot(simulated_prices.T, alpha=0.3)
        plt.title(f"Simulated Prices for {stock}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price ($)")
        plt.show()

run_model()





#123456
# cd monte-carlo
# git add .
# git commit -m "smthn"
# git push