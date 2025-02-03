import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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
