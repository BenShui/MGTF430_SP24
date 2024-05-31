import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
idx = pd.IndexSlice
deciles = np.arange(.1, 1, .1).round(1)

# Part I: Sample Selection
DATA_STORE = Path('..', 'data', 'assets.h5')

with pd.HDFStore(DATA_STORE) as store:
    data = (store['quandl/wiki/prices']
            .loc[idx['2010':'2020', :],
                 ['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
            .dropna()
            .swaplevel()
            .sort_index()
            .rename(columns=lambda x: x.replace('adj_', '')))
    metadata = store['us_equities/stocks'].loc[:, ['marketcap', 'sector']]

metadata.sector = pd.factorize(metadata.sector)[0]
data = data.join(metadata).dropna(subset=['sector'])
print(f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")
dv = data.close.mul(data.volume)
top500 = (dv.groupby(level='date')
          .rank(ascending=False)
          .unstack('ticker')
          .dropna(thresh=8*252, axis=1)
          .mean()
          .nsmallest(500))
to_drop = data.index.unique('ticker').difference(top500.index)
data = data.drop(to_drop, level='ticker')
print(f"# Tickers: {len(data.index.unique('ticker')):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")
before = len(data)
by_ticker = data.groupby(level='ticker')
T = [1, 2, 3, 4, 5, 10, 21, 42, 63, 126, 252]
for t in T:
    data[f'ret_{t:02}'] = by_ticker.close.pct_change(t)
data['ret_fwd'] = by_ticker.ret_01.shift(-1)
data = data.dropna(subset=['ret_fwd'])
data['ret'] = data.groupby('ticker').close.pct_change()
data = data[data.ret.between(-1, 1)].drop('ret', axis=1)
print(f'Dropped {before-len(data):,.0f}')
tickers = data.index.unique('ticker')
print(f"# Tickers: {len(tickers):,.0f} | # Dates: {len(data.index.unique('date')):,.0f}")
data.to_hdf('data.h5', 'data/top500')