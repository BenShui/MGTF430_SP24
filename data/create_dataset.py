import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)

DATA_STORE = Path('assets.h5')

df = (pd.read_csv('wiki_prices.csv',
                  parse_dates=['date'],
                  index_col=['date', 'ticker'],
                  infer_datetime_format=True)
      .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)


df = pd.read_csv('wiki_stocks.csv')
# no longer needed
# df = pd.concat([df.loc[:, 'code'].str.strip(),
#                 df.loc[:, 'name'].str.split('(', expand=True)[0].str.strip().to_frame('name')], axis=1)

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', df)

df = web.DataReader(name='SP500', data_source='fred', start=2010).squeeze().to_frame('close')
print(df.info())
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/fred', df)



url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url, header=0)[0]
df.columns = ['ticker', 'name', 'gics_sector', 'gics_sub_industry',
              'location', 'first_added', 'cik', 'founded']
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/stocks', df)

df = pd.read_csv('us_equities_meta_data.csv')
with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df.set_index('ticker'))