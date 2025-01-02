import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

AAPL = yf.download('AAPL', start = '2024-12-20')['Adj Close']
print(AAPL)