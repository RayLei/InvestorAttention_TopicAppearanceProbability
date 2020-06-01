#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:57:33 2020

The script download daily Google Search Volume Index
data using package 'pytrend'.

@author: LEIHAO
"""

from os import listdir
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product, chain
from scipy import stats
from functools import reduce
from itertools import product

from sqlalchemy import create_engine
from pytrends.request import TrendReq
from pytrends.dailydata import get_daily_data
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

win = 5

keywds = ['Bonds', 'Treasury Bill', 'Treasury Note', 'Treasury Bond',
          '3 month Treasury Bill', '5 Year Treasury Note',
          '10 Year Treasury Note', '30 Year Treasury Bond',
          'Healthcare Sector', 'Energy Sector', 'Crude Oil']

# keywds = ['Healthcare', 'Healthcare Industry', 'Healthcare ETF']

def get_gsvi(w):
    gsvi = get_daily_data(w, 2012, 1, 2017, 3)
    gsvi['lag_media'] = gsvi.loc[:, w].rolling(win).median().shift(1)
    gsvi['asvi'] = np.log(gsvi.loc[:, w]) - np.log(gsvi.lag_media)
    gsvi.index = gsvi.index.map(lambda x: x.date)
    return gsvi


gsvi_list = [get_gsvi(w) for w in keywds]
#out_dir = '/Users/leihao/Documents/ATAP/'
#with open(out_dir + 'p_gsvi_list.pk', 'wb') as f:
#    pk.dump(gsvi_list, f)

df_gsvi = pd.DataFrame()
for w, df in zip(keywds, gsvi_list):
    df_gsvi = pd.concat([df_gsvi, df.loc[:, [w, 'asvi']]], axis=1)

df_gsvi.columns = ['Bonds', 'B_asvi', 'Treasury Bill', 'TBil_asvi',
                   'Treasury Note', 'TN_asvi',
       'Treasury Bond', 'TBon_asvi', '3 month Treasury Bill', '3TB_asvi',
       '5 Year Treasury Note', '5TN_asvi', '10 Year Treasury Note', '10tn_asvi',
       '30 Year Treasury Note', '30Tasvi', 'Healthcare Sector', 'HC_asvi',
       'Energy Sector', 'EG_asvi', 'Crude Oil', 'CO_asvi']

df_gsvi.to_csv(out_dir + 'gsvi_daily.csv')
