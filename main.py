# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# %%
import numpy as np
import pandas as pd
from datetime import datetime

import warnings

import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rcParams['font.family'] = 'DejaVu Sans'

import scipy.stats as ss
from arch import arch_model

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use("bmh")


# %%
def raw_adj_close_prices(ticker: str, date_from: str, date_to: str):
    prices = yf.download(ticker, date_from, date_to)
    prices.index = prices.index.to_period(freq='d')
    return prices['Adj Close']


# %%
def raw_adj_close_log_returns(ticker: str, date_from: str, date_to: str):
    prices = raw_adj_close_prices(ticker, date_from, date_to)
    return np.log(prices).diff().dropna()


# %%
def beta(market: pd.Series, single_stock: pd.Series, lag: int = 252):
    return market.cov(single_stock, lag) / market.std(ddof=lag)


# %%
def arch_filtered_series(returns: pd.Series,
                         dist: str = 'skewstudent',
                         mean: str = 'HARX',
                         vol: str = 'Garch',
                         lag: int = 1,
                         p: int = 1,
                         o: int = 0,
                         q: int = 1):
    scaling_const = 10.0 / returns.std()

    model = arch_model(returns * scaling_const,
                       mean=mean, lags=lag,  # mean = Constant, ARX, HARX + the number of lags
                       vol=vol, p=p, o=o, q=q,  # vol = Garch, EGARCH, HARCH + the number of lags
                       dist=dist)  # dist = Normal, t, skewstudent, ged

    res = model.fit(update_freq=0, disp='off')
    stand_residuals = res.resid / res.conditional_volatility
    return stand_residuals, res


# %%
DATE_FROM = '2015-01-01'
DATE_TO = '2017-12-31'
spy = raw_adj_close_log_returns('SPY', DATE_FROM, DATE_TO)
spy.head()

# %%
tesla = raw_adj_close_log_returns('TSLA', DATE_FROM, DATE_TO)
tesla.head()
plt.figure(figsize=(16, 7))
tesla.plot()
plt.title("log returns")
plt.show()

# %%
filtered_tesla, model = arch_filtered_series(tesla, dist="Normal", lag=200)
plt.figure(figsize=(16, 7))
filtered_tesla.plot()
plt.title("filtered log returns")
plt.show()

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
