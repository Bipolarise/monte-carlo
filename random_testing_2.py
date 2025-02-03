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


def estimate_heston_parameters(price_data, dt):
    """
    Utilizes the Newton Raphson MLE method to estimate the parameters for the Heston Volatility model.
    
    Parameters:
    - price_data: pd.DataFrame
        Output from download_prepare_data.
    - dt: float
        Time step size.
    
    Returns:
    - heston_parameters: pd DataFrame
        DataFrame of estimated Heston parameters for each stock.
    """
    
    def log_likelihood_heston(params, log_returns, observed_v, dt):
        """
        Computes the log-likelihood of the Heston model with correlation.

        Parameters:
        - params: list
            [kappa, theta, sigma_v, rho, mu]
        - log_returns: np.array
            Logarithmic returns of the stock price
        - observed_v: np.array
            Observed variance at each time step
        - dt: float
            Time step size

        Returns:
        - log_likelihood: float
            The log-likelihood of the Heston model
        """
        kappa, theta, sigma_v, rho, mu = params
        n = len(log_returns)
        log_likelihood = 0

        for t in range(1, n):
            # Variance transition
            v_t = observed_v[t - 1]
            v_t_next = observed_v[t]
            mean_v = v_t + kappa * (theta - v_t) * dt
            var_v = sigma_v**2 * v_t * dt
            diff_v = v_t_next - mean_v
            if v_t > 0:
                log_likelihood += -0.5 * (np.log(2 * np.pi * var_v) + (diff_v**2) / var_v)

            # Stock price transition
            mean_r = (mu - 0.5 * v_t) * dt
            var_r = v_t * dt
            diff_r = log_returns[t] - mean_r
            if var_r > 0:
                # Adjust for correlation
                adjusted_var = var_r * (1 - rho**2)
                log_likelihood += -0.5 * (np.log(2 * np.pi * adjusted_var) + (diff_r**2) / adjusted_var)

        return -log_likelihood
    
    heston_parameters = {}
    v0_values = {}
    
    for stock in set(col.split("_")[0] for col in price_data.columns):
        stock_list = []
        stock_list.append(stock)
        try:
            stock_data = price_data.filter(like=stock)
            log_returns = stock_data[f"{stock}_Log Returns"]
            adj_close = stock_data[f"{stock}_Adj Close"]
            
            variance = log_returns.rolling(window=5).var()
            variance = variance.dropna()  
            
            log_returns = log_returns.loc[variance.index]  
            observed_v = variance.values  
            observed_v = np.clip(observed_v, 1e-4, None)
            
            v0 = log_returns[:15].var(ddof=1) * 252
            v0_values[stock] = max(v0, 1e-4)

            # print(f"OBSERVED VARIANCE: {observed_v[:5]}")
            
            variance_l = log_returns ** 2
            delta_variance = variance_l.diff()
            
            # print(f"Initial Variance (v0): {observed_v[0]}")

            rho = log_returns.corr(delta_variance)
            
            
            initial_guess = [
                1.0,                           # kappa: a moderate rate of reversion
                observed_v.mean(),             # theta: average variance
                observed_v.std(),              # sigma_v: standard deviation of variance
                max(observed_v[0], 1e-4),      # rho: initial variance, avoid 0
                max(min(rho, 0.9), -0.9)       # mu: clamp correlation to realistic range
            ]

            bounds = [
                (0.01, 5),    # kappa
                (0.001, 1),   # theta
                (0.01, 5),    # sigma_v
                (-1.0, 1.0),  # rho
                (-1.0, 1.0),  # mu
            ]
            
            result = differential_evolution(
                log_likelihood_heston,
                bounds=bounds,
                args=(log_returns.values, observed_v, dt),
                strategy='best1bin',
                maxiter=1000,
                
            )

            if result.success:
                heston_parameters[stock] = result.x
                

            else:
                print(f"Optimization failed for {stock}: {result.message}")
                heston_parameters[stock] = None

        
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            heston_parameters[stock] = None
    
    replace_dict = {"v0": {stock: v0 for stock, v0 in v0_values.items()}}
    
    heston_parameters = pd.DataFrame(heston_parameters)
    heston_parameters = heston_parameters.T
    heston_parameters = heston_parameters.rename(columns = {0: 'Kappa', 1: 'Theta', 2: 'Sigma_v', 3: 'Rho', 4: 'Heston Mu'})
    heston_parameters["v0"] = heston_parameters.index.map(v0_values)

    return heston_parameters

def estimate_gbm_parameters(price_data, dt):
    """
    Estimates the parameters for the Geometric Brownian Model, given the price history of the stock.
    
    Parameters:
    - price_data: pd.DataFrame
        Output from download_prepare_data.
    - dt: float
        Time step size.
        
    Returns:
    - gbm_parameters: pd.DataFrame
        DataFrame of estimated Geometric Brownian parameters for each stock.
    """
    def log_likelihood_gbm(params, log_returns, dt):
        """
        Computes the log likelihood of observing a set of parameters given the stock prices.
        """
        mu, sigma = params[0], params[1]
        N = len(log_returns)
        theoretical_mean = (mu - 0.5 * sigma**2) * dt
        
        log_likelihood = -N/2 * np.log(2 * np.pi * sigma**2 * dt) 
        log_likelihood -= 1 / (2 * sigma**2 * dt) * np.sum((log_returns - theoretical_mean)**2)
        
        return -log_likelihood        
   
    gbm_parameters = {}
    
    for stock in set(col.split("_")[0] for col in price_data.columns):
        try:
            stock_data = price_data.filter(like=stock)
            log_returns = np.log(stock_data[f"{stock}_Adj Close"] / stock_data[f"{stock}_Adj Close"].shift(1)).dropna()
            
            # print(f"Log Returns for {stock}:\n{log_returns.head()}")
            # print(f"Log Returns Standard Deviation: {log_returns.std()}")

            initial_guess = [log_returns.mean(), log_returns.std()]
            bounds = [(-np.inf, np.inf), (1e-6, np.inf)] 
            
            result = minimize(log_likelihood_gbm, initial_guess, args=(log_returns.values, dt), bounds=bounds)
            
            if result.success:
                gbm_parameters[stock] = result.x
            else:
                print(f"Optimization failed for {stock}: {result.message}")
                gbm_parameters[stock] = None
            
        except Exception as e:
            print(f"Error processing {stock}: {e}")
            gbm_parameters[stock] = None
    
    gbm_parameters = pd.DataFrame(gbm_parameters).T
    gbm_parameters.columns = ['GBM Mu', 'GBM Sigma']
    
    return gbm_parameters

def estimate_poisson_parameters(price_data, threshold_factor):
    """
    Estimates annualized Poisson parameter lambda and log-normal parameters (mu and sigma)
    directly from stock prices for a given universe.
    
    Parameters:
    - price_data: pd.DataFrame
        Output from download_prepare_data.
    - threshold_factor: float
        The threshold in standard deviations to be considered a shock.
    
    Returns:
    - poisson_parameters: pd.DataFrame
        DataFrame of estimated Poisson parameters for each stock.
    """
    
    poisson_parameters = []
    
    for stock in set(col.split("_")[0] for col in price_data.columns):
        adj_close = price_data[f"{stock}_Adj Close"]
        log_returns = price_data[f"{stock}_Log Returns"]
        S_0 = adj_close.iloc[-1]
        
        
        mu_estimate = log_returns.mean()
        sigma_estimate = log_returns.std()
        
        price_diff = adj_close.diff().dropna()
        shock_threshold = threshold_factor * price_diff.std()
        shocks = (price_diff.abs() > shock_threshold).sum()
        observation_days = len(price_diff)
        
        lambda_estimate = shocks / observation_days if observation_days > 0 else 0
        
        
        lamda_daily = lambda_estimate * (252 / observation_days) if observation_days > 0 else 0
        mu_daily = mu_estimate * 252
        sigma_daily = sigma_estimate * np.sqrt(252)
        
        poisson_parameters.append({
            'Stock': stock,
            'Lambda Daily': lamda_daily,
            'Mu Daily': mu_daily,
            'Sigma Daily': sigma_daily,
            'Shocks': shocks,
            'Initial Price': S_0
        })
        
        
    poisson_parameters = pd.DataFrame(poisson_parameters).set_index('Stock')
    
    return poisson_parameters

def all_params(price_data, dt, threshold_factor):
    """
    Combines the mu values for GBM and parameters for the Heston Volatility model
    
    Parameters:
    - universe: list
        A list of stocks to be analysed
    - start: str
        Start Date
    - end: end date
    
    Returns:
    - results_df: Pandas DataFrame
        A Pandas DataFrame containing all parameters required for the GBM and Heston Volatility
    """
    
    adj_close = price_data.filter(like='Adj Close').tail(1)
    last_stock_price = adj_close.iloc[-1].values
    
    poisson_params = estimate_poisson_parameters(price_data, threshold_factor)
    gbm_params = estimate_gbm_parameters(price_data, dt)
    heston_params = estimate_heston_parameters(price_data, dt)

    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    
    model_parameters = pd.concat([gbm_params, heston_params, poisson_params], axis = 1)
    model_parameters.reset_index(inplace = True)
    model_parameters = model_parameters.rename(columns = {'index': 'Stock'})
    
    return model_parameters

def Geometric_Brownian_Motion(S_0, mu, T, dt, n_paths, v0, kappa, theta, sigma_v):
    """
    Simulates stock prices according to the Geometric Brownian Motion.
    
    Parameters:
    - S_0     : Initial Stock Price
    - mu      : Percentage Drift
    - sigma   : Volatility.
    - T       : Terminal time
    - dt      : Time step.
    - n       : Number of simulation paths.
    - v0      : Initial Variance

    Returns:
    - stock_paths: The prices of the stocks simulated for each timestep
    """
    n_steps = int(T/dt)
    stock_paths = np.zeros((n_paths, n_steps + 1))
    stock_paths[:,0] = S_0
    
    vol_paths = simulate_heston_model(v0, kappa, theta, sigma_v, T, dt, n_paths)
    
    for i in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, size=n_paths)  # Independent noise for each path
        stock_paths[:, i] = stock_paths[:, i - 1] * np.exp(
            (mu - 0.5 * vol_paths[:, i - 1] ** 2) * dt +
            vol_paths[:, i - 1] * Z * np.sqrt(dt)
        )

    return stock_paths

def hidden_markov_model(n_steps, first_date):
    """
    Simulates the Hidden Markov Model for Market Regime Prediction. The model will output a binary variable for each day, 
    indicating whether it is high or low volatility. It will be trained with S&P 500 data. 
    
    Parameters:
    - n_steps: int
        Number of future time steps to predict
    - last_date:
        A string containing the last date of the simulation, ie the expiration date
    Returns:
    - future_regimes: pd Series
        A Series containing the regime prediction according to the hidden markov model
    """
    SP500 = yf.download('^GSPC', start = '2000-01-01', end = '2024-12-31')['Adj Close']

    SP500_returns = np.log(SP500 / SP500.shift(1)).dropna()
    X = SP500_returns.values.reshape(-1, 1)
    
    hmm = GaussianHMM(n_components = 2, covariance_type="diag", n_iter = 1000)
    hmm = hmm.fit(X)
    historical_regimes = hmm.predict(X)

    historical_regimes = pd.DataFrame(historical_regimes)
    historical_regimes = historical_regimes.set_index(SP500_returns.index)
    
    transition_matrix = hmm.transmat_
    current_state = hmm.predict(X)[-1]
    
    future_regimes = []
    for day in range(n_steps):
        next_state = np.random.choice([0,1], p = transition_matrix[current_state])
        future_regimes.append(next_state)
        current_state = next_state
        
    future_dates = pd.date_range(start=pd.to_datetime(first_date) + pd.Timedelta(days=1), periods=n_steps)
    future_regimes = pd.Series(future_regimes, index=future_dates, name='Regime')
    
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns

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

# POISSON STUFF

def simulate_stock_prices(S0, T, dt, n_paths, general_params, heston_params, gbm_params, poisson_params):
    """
    Combines the Hidden Markov Model (HMM) with the Heston volatility model, when volatility is high, and the Geometric Brownian Motion
    Model, when volatility is low.
    
    Parameters:
    S_0            : The initial stock price
    T              : The time frame in years
    dt             : The time increment
    high_vol_params: Parameters for Heston Volatility Model (obtained dynamically for each stock)
    low_vol_params : Parameters for GBM model (obtained dynamically for each stock)
    
    Returns:
    pd.DataFrame(simulated_prices): Pandas DataFrame
        A Pandas DataFrame for each simulation for the particular stock
    
    NOTE: Parameters are sourced dynamically from `all_params`.
    """
    n_steps = int(T / dt)
    print(f"N PATHS: {n_paths}")
        
    future_volatility = hidden_markov_model(n_steps, end_date)
    # print(f"Length of Future Volatility Series:{len(future_volatility)}")
    # print(f"Future Volatility Series:{future_volatility}")

    if len(future_volatility) != n_steps:
        raise ValueError(f"Length of future_volatility ({len(future_volatility)}) does not match n_steps ({n_steps}).")

    simulated_prices = np.zeros((n_paths, n_steps + 1))
    simulated_prices[:, 0] = S0
    
    # LOAD GENERAL PARAMETERS
    starting_price = general_params['Starting Price']
    stock = general_params['Stock']
    
    # LOAD POISSON PARAMETERS - All rates are annualized
    poisson_mu = poisson_params['Mu']
    poisson_lambda = poisson_params['Lambda']
    poisson_sigma = poisson_params['Sigma']
    
    # LOAD HESTON PARAMETERS
    kappa = heston_params['Kappa']
    theta = heston_params['Theta']
    sigma_v = heston_params['Sigma_v']
    v0 = heston_params['V0']
    heston_mu = heston_params['Mu']
    rho = heston_params['Rho']
    
    # LOAD GBM PARAMETERS
    gbm_mu = gbm_params['Mu']
    gbm_sigma = gbm_params['Sigma']
        
    
    # SIMULATE STOCK PRICES
    for t in range(1, n_steps + 1): 
        regime = future_volatility.iloc[t - 1]  

        # Simulate Heston model for variance
        v, _ = simulate_heston_model(starting_price, heston_mu, kappa, theta, sigma_v, v0, T, n_steps, rho, n_paths)
        vol_paths = np.sqrt(v)

        Z = np.random.normal(size=n_paths)

        if regime == "High Volatility":
            simulated_prices[:, t] = simulated_prices[:, t - 1] * np.exp(
                (-0.5 * vol_paths[:, t - 1]) * dt +
                vol_paths[:, t - 1] * Z * np.sqrt(dt)
            )
        elif regime == "Low Volatility":
            simulated_prices[:, t] = simulated_prices[:, t - 1] * np.exp(
                (gbm_mu - 0.5 * vol_paths[:, t - 1]) * dt +
                np.sqrt(vol_paths[:, t - 1]) * Z * np.sqrt(dt)
            )
        else:
            raise ValueError(f"Unexpected regime: {regime}")

    plt.figure(figsize=(12,7))
    plt.plot(simulated_prices)
    plt.title(f'Simulated Prices for {stock}')

    return pd.DataFrame(simulated_prices)
                























def run_model():
    
    global T, dt, N, threshold_factor, n_paths, end_date
    
    universe = ['AAPL', 'GOOG', 'NVDA']
    start_date, end_date = '2020-01-01', '2024-12-31'  # FOR TRAINING
    
    all_training_data = download_prepare_data(universe=universe, start=start_date, end=end_date)
    
    
    T = 0.3                 # Time to maturity in years
    dt = 1/252              # DO NOT CHANGE
    N = int(T/dt)                # DO NOT CHANGE
    threshold_factor = 1
    n_paths = 100
    
    # CALCULATE PARAMETERS FOR ALL STOCKS
    model_parameters = all_params(price_data = all_training_data, dt = dt, threshold_factor = threshold_factor)
    
    for index, row in model_parameters.iterrows():
    
        stock = row['Stock']
        kappa = row['Kappa']
        theta = row['Theta']
        sigma_v = row['Sigma_v']
        v0 = row['v0']
        heston_mu = row['Heston Mu']
        rho = row['Rho']
        gbm_mu = row['GBM Mu']
        gbm_sigma = row['GBM Sigma']
        starting_price = row['Initial Price'] 
        poisson_mu = row['Mu Daily']
        poisson_sigma = row['Sigma Daily']
        poisson_lambda = row['Lambda Daily']
    
        # ORGANIZE PARAMETERS
        general_params = {
            'Starting Price': starting_price,
            'Stock': stock
        }
        
        heston_params = {
                'Kappa': kappa,
                'Theta': theta,
                'Sigma_v': sigma_v,
                'V0': v0,
                'Mu': heston_mu,
                'Starting Price': starting_price,
                'Rho': rho
            }

        gbm_params = {
                'Mu': gbm_mu,
                'Sigma': gbm_sigma,
                
            }
            
        poisson_params = {
                'Mu': poisson_mu,
                'Sigma': poisson_sigma,
                'Lambda': poisson_lambda
            }

        simulated_prices = simulate_stock_prices(
                S0=starting_price,
                T=T,
                dt=dt,
                n_paths=n_paths,
                general_params=general_params,
                heston_params=heston_params,
                gbm_params=gbm_params,
                poisson_params = poisson_params
            )
        
        plt.figure(figsize=(12,7))
        plt.plot(simulated_prices)
        plt.title(f"Simulated Prices for {stock}")
        plt.show()
    
    
    # X = simulate_stock_prices(starting_price, T, dt, n_paths, heston_params, gbm_params, poisson_params)
    
    
    print(model_parameters)
    
run_model()


