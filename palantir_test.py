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


def heston_volatility_model(S_0, mu, kappa, theta, sigma_v, vm1, T, N, rho, prev_dW2, sm1, prev_dW1):
    
    dt = T/N
    
    dW1 = np.random.normal(0,1,N) * np.sqrt(dt)
    dZ = np.random.normal(0,1,N) * np.sqrt(dt)

    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ
    
    next_v = max(vm1 + kappa * (theta - vm1) * sigma_v * np.sqrt(vm1) * prev_dW2)
    next_s = sm1 * (1 + mu * dt + np.sqrt(vm1) * prev_dW1)

    
    
# v[i] = max(v[i-1] + kappa * (theta - v[i-1]) * dt + sigma_v * np.sqrt(v[i-1]) * dW2[i-1], 0)
