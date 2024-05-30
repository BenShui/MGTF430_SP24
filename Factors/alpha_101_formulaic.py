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