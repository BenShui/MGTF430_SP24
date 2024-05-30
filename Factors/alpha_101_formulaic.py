# Author: Jiahui Shui; Modified from: https://github.com/stefan-jansen/machine-learning-for-trading/blob/main
# /24_alpha_factor_library/03_101_formulaic_alphas.ipynb

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from talib import WMA

sns.set_style('whitegrid')
idx = pd.IndexSlice

DATA_STORE = '../data/assets.h5'

START = 2010
END = 2023

with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[str(START):str(END), :], 'adj_close']
              .unstack('ticker'))
    stocks = store['us_equities/stocks'].loc[:, ['marketcap', 'ipoyear', 'sector']]

stocks = stocks[~stocks.index.duplicated()]
stocks.index.name = 'ticker'
shared = prices.columns.intersection(stocks.index)
stocks = stocks.loc[shared, :]
prices = prices.loc[:, shared]

assert prices.shape[1] == stocks.shape[0]

# Monthly returns
monthly_prices = prices.resample('M').last()
outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                              .pct_change(lag)
                              .stack()
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                     upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1)
                              )
data = data.swaplevel().dropna()

# Drop stocks with less than 3 years of returns
min_obs = 36
nobs = data.groupby(level='ticker').size()
keep = nobs[nobs>min_obs].index

data = data.loc[idx[keep,:], :]


