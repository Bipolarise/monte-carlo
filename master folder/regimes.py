import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import yfinance as yf 



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
