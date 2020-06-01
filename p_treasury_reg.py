#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 23:41:09 2020

ATAP Topics:
    Fed Topic: 3
    Healthcare: 17
    Energy : 2, 4, 8

ASVI List:
    keywds = ['Bonds', 'Treasury Bill', 'Treasury Note', 'Treasury Bond',
              '3 month Treasury Bill', '5 Year Treasury Note',
              '10 Year Treasury Note', '30 Year Treasury Bond',
              'Healthcare Sector', 'Energy Sector', 'Crude Oil']

The script runs the regression and produce the main results in the paper.

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
from datetime import datetime, date

from sqlalchemy import create_engine
from pytrends.request import TrendReq
from pytrends.dailydata import get_daily_data
import statsmodels.api as sm
from arch import arch_model
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.anova import anova_lm
from sklearn import preprocessing


# =============================================================================
# Define functions
# =============================================================================
def read_yield(f):
    df_tmp = pd.read_csv(yield_dir + f, parse_dates=['Date'])
    df_tmp.Date = df_tmp.Date.dt.date
    df_tmp = df_tmp.set_index('Date')['Adj Close']
    df_tmp = df_tmp.dropna().to_frame('Yield')
    df_tmp['ret'] = df_tmp.Yield.diff() / df_tmp.Yield.shift(1) * 100
    return df_tmp


def get_year_week(df):
    df['year'] = df.index.map(lambda x: x.year)
    df['week'] = df.index.map(lambda x: pd.Timestamp(x).week) + 1
    df.loc[df.week == 53, 'year'] += 1
    df.loc[df.week == 53, 'week'] = 1
    df['year_week'] = df.year * 100 + df.week
    df.iloc[0, -1] = 201201
    df = df.set_index('year_week')
    return df


def get_ols_result(est):
    results_html = est.summary().tables[1].as_html()
    df_ols = pd.read_html(results_html, header=0, index_col=0)[0].iloc[:, :-2]
    df_ols.loc['adj_r2'] = est.rsquared_adj
    return df_ols


# =============================================================================
# Set parameters
# =============================================================================
win = 5
col_names = ['fed']
# gsvi_idx = 8
delta = 0
# =============================================================================
# Load the document topic matrix.
# Fed topic 3
# =============================================================================
in_dir = '/Users/leihao/Dropbox/TopicSentiment/01_mtne/'
with open(in_dir + 'p_doctpc_mtne2.pk', 'rb') as f:
    doctpc = pk.load(f)

#with open(in_dir + 'p_tpcwds_mtne2.pk', 'rb') as f:
#    tpcwds = pk.load(f)
#    
#    
#for i in np.arange(20):
#    print((doctpc[:, i] >= 1/20).sum())
# =============================================================================
# Read in the news data and symbol infor from the nasda.db
# =============================================================================
db_dir = '/Users/leihao/Dropbox/TopicSentiment/02_news/'
with open(db_dir + 'p_dates.pk', 'rb') as f:
    df_time = pk.load(f)

df_time.columns = ['date_time']
df_time = df_time.reset_index(drop=True)
df_time['timestamp'] = pd.to_datetime(df_time.date_time)
df_time['date'] = df_time.timestamp.dt.date
df_time['week'] = df_time.timestamp.dt.week
df_time['year'] = df_time.timestamp.dt.year
df_time['month'] = df_time.timestamp.dt.month

df_time.loc[(df_time.month == 12) & (df_time.week == 1), 'year'] += 1
df_time['year_week'] = df_time.year * 100 + df_time.week
df_time.loc[df_time.year_week == 201553, 'year_week'] = 201601

# =============================================================================
# Get daily atap
# =============================================================================
df_tap_time = pd.concat((pd.DataFrame(doctpc[:, 3]),
                         df_time.loc[:, 'date']),
                        axis=1)
df_tap = df_tap_time.groupby('date').sum().rename(columns={0: 'tap'})
df_tap.index = df_tap.index.rename('Date')
df_tap['lag_median'] = df_tap.rolling(win).median().shift(1)

df_tap['atap'] = np.log(df_tap.tap) - np.log(df_tap.lag_median)

# df_tap['atap_hc'] = np.log(df_tap.hc + delta) - np.log(df_tap.hc_lag + delta)
# df2013 = df_tap.loc[date(2013, 5, 1): date(2013, 8, 5)]


# =============================================================================
# Read in daily yield
# =============================================================================
yield_dir = '/Users/leihao/Downloads/'

df13_daily = read_yield('ty13_daily.csv')
df10_daily = read_yield('ty10_daily.csv')

term_spread = pd.merge(df13_daily.Yield, df10_daily.Yield, how='inner',
                       left_index=True, right_index=True)
term_spread['ts'] = term_spread.Yield_y - term_spread.Yield_x
term_spread['ts_ret'] = term_spread.ts.diff() / term_spread.ts.shift(1) * 100
# df05_daily = read_yield('ty05_daily.csv')
# df30_daily = read_yield('ty30_daily.csv')

# =============================================================================
# Read Federal Funds Rate
# =============================================================================

fedfunds_dir = '/Users/leihao/Documents/ATAP/'
fedfunds = pd.read_csv(fedfunds_dir + 'DailyFedFunds.csv', parse_dates=['DATE'])
fedfunds.DATE = pd.to_datetime(fedfunds.DATE, format='%Y-%m-%d').dt.date
fedfunds.columns = ['Date', 'volume', 'effr']
fedfunds = fedfunds.set_index('Date')
fedfunds['ff_ret'] = fedfunds.effr.diff() / fedfunds.effr.shift(1) * 100


fedfunds['dayofweek'] = fedfunds.index.map(lambda x: pd.Timestamp(x).weekday())
ff_weekly = fedfunds[fedfunds.dayofweek == 4].sort_index()
ff_weekly['ff_ret'] = ff_weekly.effr.diff() / ff_weekly.effr.shift(1) * 100
ff_weekly = ff_weekly.loc[date(2012, 1, 6): date(2017, 4, 1)]
ff_weekly = get_year_week(ff_weekly)
ff_weekly = ff_weekly.loc[:, ['effr', 'ff_ret']].rename(columns={'effr': 'ff'})



# =============================================================================
# Get GSVI Daily
# =============================================================================
gsvi_dir = '/Users/leihao/Documents/ATAP/'
with open(gsvi_dir + 'p_gsvi_list.pk', 'rb') as f:
    gsvi_list = pk.load(f)

df_gsvi_bond = gsvi_list[0]
df_gsvi_bond['lag_median'] = df_gsvi_bond.Bonds.rolling(win).median().shift(1)
df_gsvi_bond.asvi = np.log(df_gsvi_bond.loc[:, 'Bonds'] + delta) -\
                    np.log(df_gsvi_bond.lag_media + delta)


#df_gsvi_tbil = gsvi_list[1]
## df_gsvi_tbil['lag_media'] = df_gsvi_tbil.loc[:, 'Treasury Bill']\
##                                         .rolling(win).median().shift(1)
## test = df_gsvi_tbil.loc[:, 'Treasury Bill'].rolling(win).median().shift(1)
#df_gsvi_tbil.asvi = np.log(df_gsvi_tbil.loc[:, 'Treasury Bill'] + delta) -\
#                    np.log(df_gsvi_tbil.lag_media + delta)
#
#df_gsvi_tnot = gsvi_list[2]
#df_gsvi_tnot.asvi = np.log(df_gsvi_tnot.loc[:, 'Treasury Note'] + delta) -\
#                    np.log(df_gsvi_tnot.lag_media + delta)
#
#df_gsvi_tbon = gsvi_list[3]
#df_gsvi_tbon.asvi = np.log(df_gsvi_tbon.loc[:, 'Treasury Bond'] + delta) -\
#                    np.log(df_gsvi_tbon.lag_media + delta)






#fig, (ax1, ax2) = plt.subplots(2)
## fig.suptitle('Vertically stacked subplots')
#color = 'tab:orange'
#ax1.set_xlabel('Date ')
#ax1.set_ylabel('Daily', color=color)
#ax1.plot_date(gsvi_daily.index, gsvi_daily.Bonds, color=color, fmt='--')
#ax1.tick_params(axis='y', labelcolor=color)
#
#color = 'tab:blue'
#ax2.set_ylabel('Weekly', color=color)  # we already handled the x-label with ax1
#ax2.plot_date(gsvi.Date, gsvi.gsvi, color=color, fmt='-')
#ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#fig.savefig(out_dir + 'svi.png', dpi=400)


# =============================================================================
#  Merge: 13 week T-Bill
# =============================================================================
tmp1 = pd.merge(df_tap.tap, df_gsvi_bond.Bonds, how='inner',
                left_index=True, right_index=True)
df13_daily = df13_daily[df13_daily.ret < 1000]
df1 = pd.merge(tmp1, df13_daily.ret, how='inner',
               left_index=True, right_index=True)
df1 = pd.merge(df1, fedfunds.ff_ret, how='inner',
               left_index=True, right_index=True)


def get_abnormal(s, scale=True):
    lag_median = s.rolling(win).median().shift(1)
    abn_values = np.log(s) - np.log(lag_median)
    if scale:
        abn_values = pd.Series(preprocessing.scale(abn_values), index=s.index)
    return abn_values

df1['atap'] = get_abnormal(df1.tap)
df1['asvi'] = get_abnormal(df1.Bonds)
df1 = df1.drop(index=[date(2013, 6, 24), date(2013, 7, 30)]).dropna()

df1 = df1.drop(columns=['tap', 'Bonds'])






tmp1 = pd.merge(df_tap.loc[:, ['atap']], df_gsvi_bond.asvi, how='inner',
                left_index=True, right_index=True)

df1 = pd.merge(tmp1, df13_daily.Yield.diff().rename('ret'), how='inner',
               left_index=True, right_index=True)


df1 = pd.merge(df1, fedfunds.loc[:, 'effr'].diff().rename('ff_ret'), how='inner',
               left_index=True, right_index=True)

df1 = df1.drop(index=[date(2013, 6, 24), date(2013, 7, 30)]).dropna()
df1 = df1.loc[df1.atap > -4]


df1_scaled = pd.DataFrame(preprocessing.scale(df1), index=df1.index,
                          columns=df1.columns)

df1 = df1.loc[df1.ret.abs() < 100]

#df1 = pd.merge(df1, fedfunds.loc[:, ['effr' 'ff_ret']], how='inner',
#               left_index=True, right_index=True)
#
#df1 = pd.merge(df1, term_spread.loc[:, ['ts', 'ts_ret']], how='inner',
#               left_index=True, right_index=True)

# df1 = df1.dropna()
# df1['y'] = df1.ret.shift(-1)
# Very few articles in 2013-6-24 and 2017-7-30
# no articles between these two dates
# atap is outerliers for these two days



#df1 = df1.drop(index=[date(2013, 6, 24), date(2013, 7, 30),
#                      date(2016, 12, 5)]).dropna()

# =============================================================================
# Granger causality test
# =============================================================================
grangercausalitytests(df1.loc[:, ['atap', 'asvi']], 5)
grangercausalitytests(df1.loc[:, ['asvi', 'atap']], 5)

# =============================================================================
# Impulse response - daily
# =============================================================================
df1_tmp = df1.loc[:, ['asvi', 'atap', 'ret']]

results = VAR(df1_tmp).fit(5)
# results.plot()
# results.plot_acorr()

irf = results.irf(10)
fig = irf.plot(orth=True)
out_dir = '/Users/leihao/Documents/ATAP/'
fig.savefig(out_dir + 'daily13_impulse_final_norm_nooutlier.png', dpi=400)


# =============================================================================
# Merge: 10 year T-Note
# =============================================================================
tmp1 = pd.merge(df_tap.tap, df_gsvi_bond.Bonds, how='inner',
                left_index=True, right_index=True)
df1 = pd.merge(tmp1, df10_daily.ret, how='inner',
               left_index=True, right_index=True)
df1 = pd.merge(df1, term_spread.loc[:, ['ts', 'ts_ret']], how='inner',
               left_index=True, right_index=True)


def get_abnormal(s, scale=True):
    lag_median = s.rolling(win).median().shift(1)
    abn_values = np.log(s) - np.log(lag_median)
    if scale:
        abn_values = pd.Series(preprocessing.scale(abn_values), index=s.index)
    return abn_values

df1['atap'] = get_abnormal(df1.tap)
df1['asvi'] = get_abnormal(df1.Bonds)
df1 = df1.drop(index=[date(2013, 6, 24), date(2013, 7, 30)]).dropna()

df1 = df1.drop(columns=['tap', 'Bonds'])





tmp1 = pd.merge(df_tap.loc[:, ['atap']], df_gsvi_bond.asvi, how='inner',
                left_index=True, right_index=True)

df1 = pd.merge(tmp1, df10_daily.ret, how='inner',
               left_index=True, right_index=True)

df1 = pd.merge(df1, term_spread.loc[:, ['ts', 'ts_ret']], how='inner',
               left_index=True, right_index=True)

df1 = df1.drop(index=[date(2013, 6, 24), date(2013, 7, 30)]).dropna()



# =============================================================================
# Granger causality test
# =============================================================================
grangercausalitytests(df1.loc[:, ['atap_fed', 'asvi']], 5)
grangercausalitytests(df1.loc[:, ['asvi', 'atap_fed']], 5)

# =============================================================================
# Impulse response - daily
# =============================================================================
df1_tmp = df1.loc[:, ['asvi', 'atap', 'ret']]
results = VAR(df1_tmp).fit(5)
# results.plot()
# results.plot_acorr()

irf = results.irf(10)
fig = irf.plot(orth=True)
fig.savefig(out_dir + 'daily10_impulse_final_norm.png', dpi=400)


# =============================================================================
# Get the product term
# =============================================================================
#df1['ret_sig'] = (df1.ret < 0).astype('int')
#df1['atap_sig_ret'] = df1.loc[:, ['ret_sig', 'atap_fed']].prod(axis=1)
#df1['asvi_sig_ret'] = df1.loc[:, ['ret_sig', 'asvi']].prod(axis=1)
#df1 = df1.drop(columns=['ret_sig'])

df1['atap_ret'] = df1.loc[:, ['ret', 'atap']].prod(axis=1)
df1['asvi_ret'] = df1.loc[:, ['ret', 'asvi']].prod(axis=1)
#df1 = df1.dropna()
#df1.shape

# =============================================================================
# t - t
# =============================================================================
df1['ret_l1'] = df1.ret.shift(1)
df1 = df1.dropna()
exp1 = sm.OLS(df1.ret, sm.add_constant(df1.loc[:, ['asvi', 'ret_l1']])).fit()
exp2 = sm.OLS(df1.ret, sm.add_constant(df1.loc[:, ['atap', 'ret_l1']])).fit()
exp3 = sm.OLS(df1.ret, sm.add_constant(df1.loc[:, ['atap', 'asvi', 'ret_l1']])).fit()




ols_exp1 = get_ols_result(exp1)
ols_exp2 = get_ols_result(exp2)
ols_exp3 = get_ols_result(exp3)


# =============================================================================
# Dataframe: 13 weeks
# =============================================================================
df13_exp = pd.merge(ols_exp1, ols_exp2, how='outer',
                    left_index=True, right_index=True)
df13_exp = pd.merge(df13_exp, ols_exp3, how='outer',
                    left_index=True, right_index=True)

# =============================================================================
# Dataframe: 10 years
# =============================================================================
df10_exp = pd.merge(ols_exp1, ols_exp2, how='outer',
                    left_index=True, right_index=True)
df10_exp = pd.merge(df10_exp, ols_exp3, how='outer',
                    left_index=True, right_index=True)

df_exp = pd.concat([df13_exp, df10_exp], axis=1)
out_dir = '/Users/leihao/Documents/ATAP/'
df_exp.to_csv(out_dir + 'p_treasury_ret_exp_daily_results_norm.csv')


# =============================================================================
# t+1 ~ lag values
# =============================================================================
# df1 = df1[df1.ret < 800]
tmp = pd.concat([df1.shift(i) for i in np.arange(5)], axis=1)
cols_tmp = [[c + str(i) for c in df1.columns] for i in np.arange(5)]
tmp.columns = list(chain(*cols_tmp))
tmp['y'] = tmp.ret0.shift(-1)
# tmp['y'] = tmp.atap0.shift(-1)
# tmp['y'] = tmp.asvi0.shift(-1)
df_pred = tmp.dropna()
#df_pred = df_pred[df_pred < 800 ].dropna()
cols = df_pred.columns[:-1]
cols1 = [c for c in cols if 'atap' not in c]
cols2 = [c for c in cols if 'asvi' not in c]

#cols1_sub = [c for c in cols1 if 'ff_ret' not in c and 'effr' not in c]
#cols2_sub = [c for c in cols2 if 'ff_ret' not in c and 'effr' not in c]

pred1 = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols1])).fit()
pred1.summary()

#pred1_sub = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols1_sub])).fit()
#res1 = anova_lm(pred1_sub, pred1)
#print(res1)

pred2 = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols2])).fit()
pred2.summary()

#pred2_sub = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols2_sub])).fit()
#res2 = anova_lm(pred2_sub, pred2)
#print(res2)

pred3 = sm.OLS(df_pred.y, sm.add_constant(df_pred.iloc[:, :-1])).fit()
pred3.summary()

ols_pred1 = get_ols_result(pred1)
ols_pred2 = get_ols_result(pred2)
ols_pred3 = get_ols_result(pred3)

# =============================================================================
# Dataframe: 13 weeks
# =============================================================================
df13_ols = pd.merge(ols_pred1, ols_pred2, how='outer',
                    left_index=True, right_index=True)
df13_ols = pd.merge(df13_ols, ols_pred3, how='outer',
                    left_index=True, right_index=True)
out_dir = '/Users/leihao/Documents/ATAP/'
df13_ols.to_csv(out_dir + 'p_3m_nooutlier_ret_pred_daily_results_norm_win10.csv')


# =============================================================================
# Dataframe: 10 year
# =============================================================================
df10_ols = pd.merge(ols_pred1, ols_pred2, how='outer',
                    left_index=True, right_index=True)
df10_ols = pd.merge(df10_ols, ols_pred3, how='outer',
                    left_index=True, right_index=True)

df_daily_out = pd.merge(df13_ols, df10_ols, how='outer',
                        left_index=True, right_index=True)

df_daily_out.to_csv(out_dir + 'p_treasury_ret_pred_daily_results_final_norm_win10.csv')



# =============================================================================
# Weekly
# =============================================================================
week_win = 8
# =============================================================================
# Read in weekly yields
# =============================================================================

df13_weekly = read_yield('ty13_weekly.csv')
df10_weekly = read_yield('ty10_weekly.csv')
# In the weekly data, Sunday is the start of the week.
# 2012-01-08 is actually the yield at date 2012-01-13, Friday
# However 2012-01-08 is classified as week 1 in pandas


df13_weekly = get_year_week(df13_weekly)
df10_weekly = get_year_week(df10_weekly)

ts_weekly = (df10_weekly.Yield - df13_weekly.Yield).rename('ts').to_frame()
ts_weekly['ts_ret'] = ts_weekly.ts.diff() / ts_weekly.ts.shift(1) * 100

#df05_weekly = get_year_week(df05_weekly)
#df30_weekly = get_year_week(df30_weekly)
#
#fig, (ax1, ax2) = plt.subplots(2)
## fig.suptitle('Vertically stacked subplots')
#color = 'tab:blue'
#ax1.set_xlabel('Date ')
#ax1.set_ylabel('3 Month', color=color)
#ax1.plot_date(df13_daily.index, df13_daily.Yield.values, color=color, fmt='--')
#ax1.tick_params(axis='y', labelcolor=color)
#
#color = 'tab:orange'
#ax2.set_ylabel('10 Year', color=color)  # we already handled the x-label with ax1
#ax2.plot_date(df10_daily.index, df10_daily.Yield.values, color=color, fmt='-')
#ax2.tick_params(axis='y', labelcolor=color)
#
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()
#
#fig.savefig(out_dir + 'yields.png', dpi=400)


# =============================================================================
# Get GSVI Weekly
# =============================================================================
df_bonds_weekly = pd.read_csv(gsvi_dir + 'Bonds_weekly.csv',
                              parse_dates=['Date'], dayfirst=True)
df_bonds_weekly.Bonds.describe()

#df_bonds_weekly2 = df_bonds_weekly.copy()
#df_bonds_weekly2['year'] = df_bonds_weekly2.Date.dt.year
#df_bonds_weekly2['week'] = df_bonds_weekly2.Date.dt.week + 1 # in df_bonds_weekly, date is week start date
#df_bonds_weekly2.loc[df_bonds_weekly2.week == 53, 'year'] += 1
#df_bonds_weekly2.loc[df_bonds_weekly2.week == 53, 'week'] = 1
#df_bonds_weekly2['month'] = df_bonds_weekly2.Date.dt.month
#df_bonds_weekly2['lag_median'] = df_bonds_weekly2.Bonds.rolling(week_win).median().shift(1)
#df_bonds_weekly2['asvi'] = np.log(df_bonds_weekly2.Bonds) - np.log(df_bonds_weekly2.lag_median)
#df_bonds_weekly2['year_week'] = df_bonds_weekly2.year * 100 + df_bonds_weekly2.week
#df_bonds_weekly2 = df_bonds_weekly2.set_index('year_week')



df_bonds_weekly['year_week'] = df_bonds_weekly.Date.apply(lambda x: x.year * 100 + x.week + 1)
df_bonds_weekly.loc[0, 'year_week'] = 201201
df_bonds_weekly.loc[52, 'year_week'] = 201301
df_bonds_weekly.loc[104, 'year_week'] = 201401
df_bonds_weekly.loc[156, 'year_week'] = 201501
df_bonds_weekly.loc[208, 'year_week'] = 201601
df_bonds_weekly.loc[209, 'year_week'] = 201601
df_bonds_weekly.loc[261, 'year_week'] = 201701
df_bonds_weekly = df_bonds_weekly.set_index('year_week')\
                                 .drop(columns=['Date'])
# df_bonds_weekly.index = df_bonds_weekly.index.rename('Date')
df_bonds_weekly['lag_median'] = df_bonds_weekly.rolling(week_win).median().shift(1)
df_bonds_weekly['asvi'] = np.log(df_bonds_weekly.Bonds) - np.log(df_bonds_weekly.lag_median)



# =============================================================================
# Get weekly atap
# =============================================================================

df_tap_date = pd.concat((pd.DataFrame(doctpc[:, 3]),
                         df_time.loc[:, 'year_week']),
                        axis=1)
df_tap_week = df_tap_date.groupby('year_week').sum().rename(columns={0: 'tap'})
# df_tap_week.index = df_tap_week.index.rename('Date')
df_tap_week['lag_median'] = df_tap_week.rolling(week_win).median().shift(1)
df_tap_week['atap'] = np.log(df_tap_week.tap) - np.log(df_tap_week.lag_median)
# df_tap_week = df_tap_week.drop(index=201326)


# =============================================================================
# Merge 3 month weekly 
# =============================================================================
tmp_w1 = pd.merge(df_tap_week.atap, df_bonds_weekly.asvi, how='right',
                  left_index=True, right_index=True)

# remove the outlier
tmp_w1 = tmp_w1[tmp_w1 > -6].dropna()

tmp_w1 = pd.DataFrame(preprocessing.scale(tmp_w1), index=tmp_w1.index,
                      columns=tmp_w1.columns)

# 13-week
#tmp_w2 = pd.merge(tmp_w1, df13_weekly.ret, how='inner',
#                  left_index=True, right_index=True)
#
#tmp_w2['ret_l1'] = tmp_w2.ret.shift(1)
#tmp_w2 = tmp_w2.dropna()

df13_weekly = df13_weekly[df13_weekly.ret < 1000]

tmp_w2 = pd.merge(tmp_w1, df13_weekly.ret, how='inner',
                  left_index=True, right_index=True)
tmp_w2 = pd.merge(tmp_w2, ff_weekly.ff_ret, how='inner',
                  left_index=True, right_index=True)




# 10-year
tmp_w2 = pd.merge(tmp_w1, df10_weekly.ret, how='left',
                  left_index=True, right_index=True)
tmp_w2 = pd.merge(tmp_w2, ts_weekly, how='left',
                  left_index=True, right_index=True)
tmp_w2 = tmp_w2.dropna()

#tmp_w3['ret_l1'] = tmp_w3.ret.shift(1)
#tmp_w3 = tmp_w3.dropna()
# =============================================================================
# Impulse response - weekly
# =============================================================================
tmp_w3 = tmp_w2.loc[:, ['asvi', 'atap', 'ret']]
results = VAR(tmp_w3).fit(5)
# results.plot()
# results.plot_acorr()

irf = results.irf(10)
fig = irf.plot(orth=True)
out_dir = '/Users/leihao/Documents/ATAP/'
fig.savefig(out_dir + 'weekly13_impulse_final_norm.png', dpi=400)



# =============================================================================
# weekly t-t regression
# =============================================================================
tmp_w2['ret_l1'] = tmp_w2.ret.shift(1)
tmp_w2 = tmp_w2.dropna()

exp_w1 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['asvi', 'ret_l1']])).fit()
exp_w2 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['atap', 'ret_l1']])).fit()
exp_w3 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['atap', 'asvi', 'ret_l1']])).fit()

ols_exp_w1 = get_ols_result(exp_w1)
ols_exp_w2 = get_ols_result(exp_w2)
ols_exp_w3 = get_ols_result(exp_w3)

exp_w4 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['asvi', 'ret_l1']])).fit()
exp_w5 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['atap', 'ret_l1']])).fit()
exp_w6 = sm.OLS(tmp_w2.ret, sm.add_constant(tmp_w2.loc[:, ['atap', 'asvi', 'ret_l1']])).fit()

ols_exp_w4 = get_ols_result(exp_w4)
ols_exp_w5 = get_ols_result(exp_w5)
ols_exp_w6 = get_ols_result(exp_w6)

df_exp_w = reduce(lambda x, y: pd.merge(x, y, how='outer',
                                        left_index=True, right_index=True),
                 [ols_exp_w1, ols_exp_w2, ols_exp_w3, ols_exp_w4, ols_exp_w5,
                  ols_exp_w6])
out_dir = '/Users/leihao/Documents/ATAP/'
df_exp_w.to_csv(out_dir + 'p_treasury_ret_exp_week_results_norm.csv')


# =============================================================================
# Product term
# =============================================================================
tmp_w2['atap_ret'] = tmp_w2.loc[:, ['ret', 'atap']].prod(axis=1)
tmp_w2['asvi_ret'] = tmp_w2.loc[:, ['ret', 'asvi']].prod(axis=1)


# =============================================================================
# Weekly t ~ t-1 ols
# =============================================================================
# tmp_w3 = tmp_w2.loc[(tmp_w2.atap > -4) & (tmp_w2.ret < 800)]

tmp = pd.concat([tmp_w2.shift(i) for i in np.arange(5)], axis=1)
cols_tmp = [[c + str(i) for c in tmp_w2.columns] for i in np.arange(5)]
tmp.columns = list(chain(*cols_tmp))
tmp['y'] = tmp.ret0.shift(-1)
# tmp['y'] = tmp.atap0.shift(-1)
# tmp['y'] = tmp.asvi0.shift(-1)
df_pred = tmp.dropna()
# df_pred = df_pred.loc[:201706]
# df_pred = df_pred[df_pred > -4 ].dropna()
cols = df_pred.columns[:-1]
cols1 = [c for c in cols if 'atap' not in c]
cols2 = [c for c in cols if 'asvi' not in c]

#cols1_sub = [c for c in cols1 if 'ff_ret' not in c and 'effr' not in c]
#cols2_sub = [c for c in cols2 if 'ff_ret' not in c and 'effr' not in c]
#cols2_sub = ['atap0', 'ret0']

pred1 = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols1])).fit()
pred1.summary()

#pred1_sub = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols1_sub])).fit()
#res1 = anova_lm(pred1_sub, pred1)
#print(res1)

pred2 = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols2])).fit()
pred2.summary()

#pred2_sub = sm.OLS(df_pred.y, sm.add_constant(df_pred.loc[:, cols2_sub])).fit()
#res2 = anova_lm(pred2_sub, pred2)
#print(res2)

pred3 = sm.OLS(df_pred.y, sm.add_constant(df_pred.iloc[:, :-1])).fit()
pred3.summary()

# for 3-month
ols_pred1 = get_ols_result(pred1)
ols_pred2 = get_ols_result(pred2)
ols_pred3 = get_ols_result(pred3)

df_pred_w = reduce(lambda x, y: pd.merge(x, y, how='outer',
                                        left_index=True, right_index=True),
                 [ols_pred1, ols_pred2, ols_pred3])
out_dir = '/Users/leihao/Documents/ATAP/'
df_pred_w.to_csv(out_dir + 'p_3m_nooutlier_ret_pred_week_results_norm_win12.csv')



# for 10-year
ols_pred4 = get_ols_result(pred1)
ols_pred5 = get_ols_result(pred2)
ols_pred6 = get_ols_result(pred3)

df_pred_w = reduce(lambda x, y: pd.merge(x, y, how='outer',
                                        left_index=True, right_index=True),
                 [ols_pred1, ols_pred2, ols_pred3, ols_pred4, ols_pred5,
                  ols_pred6])
out_dir = '/Users/leihao/Documents/ATAP/'
df_pred_w.to_csv(out_dir + 'p_treasury_ret_pred_week_results_final_norm_win12.csv')


# =============================================================================
# Monthly volatility model
# =============================================================================

# get the monthly volatility
df13_daily['year_month'] = df13_daily.index.map(lambda x: x.year * 100 + x.month)
df10_daily['year_month'] = df10_daily.index.map(lambda x: x.year * 100 + x.month)

df13_daily.ret /= 100
df10_daily.ret /= 100

df13_vol_month = df13_daily.groupby('year_month')['ret'].std().rename('vol')
df10_vol_month = df10_daily.groupby('year_month')['ret'].std().rename('vol')

# get the monthly atap
win = 5
df_fed_time = pd.concat((pd.DataFrame(doctpc[:, 3]), df_time.loc[:, 'date']), axis=1)
df_fed_time['year_month'] = df_fed_time.date.apply(lambda x: x.year * 100 + x.month)
df_fed_month = df_fed_time.groupby('year_month').sum()
df_fed_month['lag_median'] = df_fed_month.rolling(win).median().shift(1)
df_fed_month['atap'] = np.log(df_fed_month.loc[:, 0]) - np.log(df_fed_month.lag_median)
df_fed_month.atap = pd.Series(preprocessing.scale(df_fed_month.atap),
                              index=df_fed_month.atap.index)

#median = df_fed_month.atap.median()
#df_fed_month.loc[df_fed_month.atap < -2, 'atap'] = median

# get the monthly asvi
gsvi_dir = '/Users/leihao/Downloads/'
gsvi_month = pd.read_csv(gsvi_dir + 'Bonds_monthly.csv', parse_dates=['Month'])
gsvi_month.columns = ['Date', 'gsvi']
gsvi_month['year_month'] = gsvi_month.Date.apply(lambda x: x.year * 100 + x.month)
gsvi_month['lag_median'] = gsvi_month.gsvi.rolling(win).median().shift(1)
gsvi_month['asvi'] = np.log(gsvi_month.gsvi) - np.log(gsvi_month.lag_median)
gsvi_month = gsvi_month.set_index('year_month')
gsvi_month.asvi = pd.Series(preprocessing.scale(gsvi_month.asvi),
                            index=gsvi_month.asvi.index)

# read in the monthly yield data
my_dir = '~/Downloads/'

def get_monthly_return(f):
    df_tmp = pd.read_csv(my_dir + f, parse_dates=['Date'])
    df_tmp['ret'] = df_tmp.loc[:, 'Adj Close'].diff() / df_tmp.loc[:, 'Adj Close'].shift(1)
    df_tmp['year_month'] = df_tmp.Date.apply(lambda x: x.year * 100 + x.month)
    df_tmp = df_tmp.set_index('year_month')
    return df_tmp.ret

df13_ret_month = get_monthly_return('ty13_monthly.csv')
df10_ret_month = get_monthly_return('ty10_monthly.csv')

#df13_vol_boxcox = pd.DataFrame(df13_vol_month,
#                               index=df13_vol_month.index,
#                               columns=['vol'])
# Merge data
df13_month = pd.concat([df13_vol_month, df13_ret_month,
                        df_fed_month.atap, gsvi_month.asvi], axis=1)
df10_month = pd.concat([df10_vol_month, df10_ret_month,
                        df_fed_month.atap, gsvi_month.asvi], axis=1)

def get_xy(df):
    df.columns = ['V', 'R', 'T', 'G']
    df['RT'] = df.loc[:, ['R', 'T']].prod(axis=1)
    df['RG'] = df.loc[:, ['R', 'G']].prod(axis=1)
    df_tmp = pd.concat([df.shift(i) for i in np.arange(5)], axis=1)
    cols_tmp4 = [[col + str(i) for col in df.columns] for i in np.arange(5)]
    df_tmp.columns = list(chain(*cols_tmp4))
    df_tmp['y'] = df_tmp.V0.shift(-1)
    return df_tmp

df13_xy = get_xy(df13_month).dropna()
df10_xy = get_xy(df10_month).dropna()

cols1 = [c for c in df13_xy.iloc[:, :-1].columns if 'T' not in c]
cols2 = [c for c in df13_xy.iloc[:, :-1].columns if 'G' not in c]
cols3 = [c for c in df13_xy.iloc[:, :-1].columns if 'y' not in c]

vol_month1 = sm.OLS(df13_xy.y, sm.add_constant(df13_xy.loc[:, cols1])).fit()
vol_month2 = sm.OLS(df13_xy.y, sm.add_constant(df13_xy.loc[:, cols2])).fit()
vol_month3 = sm.OLS(df13_xy.y, sm.add_constant(df13_xy.loc[:, cols3])).fit()

vol_month4 = sm.OLS(df10_xy.y, sm.add_constant(df10_xy.loc[:, cols1])).fit()
vol_month5 = sm.OLS(df10_xy.y, sm.add_constant(df10_xy.loc[:, cols2])).fit()
vol_month6 = sm.OLS(df10_xy.y, sm.add_constant(df10_xy.loc[:, cols3])).fit()

ols_vol_month1 = get_ols_result(vol_month1)
ols_vol_month2 = get_ols_result(vol_month2)
ols_vol_month3 = get_ols_result(vol_month3)

ols_vol_month4 = get_ols_result(vol_month4)
ols_vol_month5 = get_ols_result(vol_month5)
ols_vol_month6 = get_ols_result(vol_month6)


df_vol_month = pd.merge(ols_vol_month1, ols_vol_month2, how='outer',
                        left_index=True, right_index=True)
df_vol_month = pd.merge(df_vol_month, ols_vol_month3, how='outer',
                        left_index=True, right_index=True)
df_vol_month = pd.merge(df_vol_month, ols_vol_month4, how='outer',
                        left_index=True, right_index=True)
df_vol_month = pd.merge(df_vol_month, ols_vol_month5, how='outer',
                        left_index=True, right_index=True)
df_vol_month = pd.merge(df_vol_month, ols_vol_month6, how='outer',
                        left_index=True, right_index=True)
out_dir = '/Users/leihao/Documents/ATAP/'
df_vol_month.to_csv(out_dir + 'monthly_vol_pred_results_win' + str(win) + '_norm.csv')


pcols = [c for c in df_vol_month if 'P>' in c]
df_vol_month.loc[:, pcols]

# =============================================================================
# Volatility Model GARCH(1, 1)
# =============================================================================

# =============================================================================
# Get X
# =============================================================================
tmp_daily = pd.merge(df_tap.loc[:, ['atap']], df_gsvi_bond.asvi, how='inner',
                     left_index=True, right_index=True)
tmp_daily = pd.DataFrame(preprocessing.scale(tmp_daily),
                         index=tmp_daily.index, columns=tmp_daily.columns)
df_att_daily = pd.concat([tmp_daily.shift(i) for i in np.arange(6)], axis=1)
cols_tmp = [[c + str(i) for c in tmp_daily.columns] for i in np.arange(6)]
df_att_daily.columns = list(chain(*cols_tmp))

tmp_weekly = pd.merge(df_tap_week.atap, df_bonds_weekly.asvi, how='right',
                      left_index=True, right_index=True)
tmp_weekly = pd.DataFrame(preprocessing.scale(tmp_weekly),
                          index=tmp_weekly.index, columns=tmp_weekly.columns)
df_att_weekly = pd.concat([tmp_weekly.shift(i) for i in np.arange(6)], axis=1)
cols_tmp = [[c + str(i) for c in tmp_weekly.columns] for i in np.arange(6)]
df_att_weekly.columns = list(chain(*cols_tmp))

#df_pred = tmp.dropna()
#cols = df_pred.columns[:-1]
#cols1 = [c for c in cols if 'atap' not in c]
#cols2 = [c for c in cols if 'asvi' not in c]


def get_volatility(df):
    garch11 = arch_model(df, p=1, q=1)
    res = garch11.fit(update_freq=10)
    print(res.summary())
    plt.plot(res.conditional_volatility.values)
    return(res.conditional_volatility)

df13_daily.ret *= 100
df10_daily.ret *= 100

df10_daily_vol = get_volatility(df10_daily.ret.dropna() ) ** 2
df10_weekly_vol = get_volatility(df10_weekly.ret.dropna() ) ** 2

# remove outliers
tmp_vol_weekly = df13_weekly.loc[df13_weekly.ret < 500, 'ret'].dropna() 
tmp_vol_daily = df13_daily.loc[df13_daily.ret < 500, 'ret'].dropna()

tmp_vol_weekly /= 100
tmp_vol_daily /= 100

df13_daily_vol = get_volatility(tmp_vol_daily) ** 2
df13_weekly_vol = get_volatility(tmp_vol_weekly) ** 2

def get_vol_lag_value(vol):
    tmp2 = pd.concat([vol.shift(i) for i in np.arange(2)], axis=1)
    tmp2.columns = [ 'vol' + str(i) for i in np.arange(2)]
    # tmp2['y'] = vol.shift(-1)
    # tmp2.columns = list(chain(*cols_tmp))
    return tmp2

df13_daily_vol_lag = get_vol_lag_value(df13_daily_vol)
df13_weekly_vol_lag = get_vol_lag_value(df13_weekly_vol)
df10_daily_vol_lag = get_vol_lag_value(df10_daily_vol)
df10_weekly_vol_lag = get_vol_lag_value(df10_weekly_vol)

df13_daily_vol_pred = pd.merge(df_att_daily, df13_daily_vol_lag,
                               how='inner', left_index=True, right_index=True)\
                        .dropna()
df10_daily_vol_pred = pd.merge(df_att_daily, df10_daily_vol_lag,
                               how='inner', left_index=True, right_index=True)\
                        .dropna()
df13_weekly_vol_pred = pd.merge(df_att_weekly, df13_weekly_vol_lag,
                                how='inner', left_index=True, right_index=True)\
                        .dropna()
df10_weekly_vol_pred = pd.merge(df_att_weekly, df10_weekly_vol_lag,
                                how='inner', left_index=True, right_index=True)\
                         .dropna()                       
                        
# cols1 = [c for c in df13_daily_vol_pred.iloc[:, :-1].columns if 'atap' in c]
# cols2 = [c for c in df13_daily_vol_pred.iloc[:, :-1].columns if 'asvi' in c]


def get_vol_pred(Y, X):
    res = sm.OLS(Y, X).fit()
    print(res.summary())
    return res

# =============================================================================
# 3 month daily
# =============================================================================

cols_vol4 = ['asvi' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol5 = ['atap' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol6 = (cols_vol4 + cols_vol5)[:-1]

# df13_daily_vol_pred = df13_daily_vol_pred.loc[(df13_daily_vol_pred.iloc[:, :-2] > -4).all(1)]

X13_asvi = sm.add_constant(df13_daily_vol_pred.loc[:, cols_vol4])
X13_atap = sm.add_constant(df13_daily_vol_pred.loc[:, cols_vol5])
X13_alll = sm.add_constant(df13_daily_vol_pred.loc[:, cols_vol6])
Y13 = df13_daily_vol_pred.vol0
vol_daily1 = get_ols_result(get_vol_pred(Y13, X13_asvi))
vol_daily2 = get_ols_result(get_vol_pred(Y13, X13_atap))
vol_daily3 = get_ols_result(get_vol_pred(Y13, X13_alll))

# =============================================================================
# 10 year daily
# =============================================================================
cols_vol4 = ['asvi' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol5 = ['atap' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol6 = (cols_vol4 + cols_vol5)[:-1]
X10_asvi = sm.add_constant(df10_daily_vol_pred.loc[:, cols_vol4])
X10_atap = sm.add_constant(df10_daily_vol_pred.loc[:, cols_vol5])
X10_alll = sm.add_constant(df10_daily_vol_pred.loc[:, cols_vol6])
Y10=df10_daily_vol_pred.vol0
vol_daily4 = get_ols_result(get_vol_pred(Y10, X10_asvi))
vol_daily5 = get_ols_result(get_vol_pred(Y10, X10_atap))
vol_daily6 = get_ols_result(get_vol_pred(Y10, X10_alll))

df_vol_daily = reduce(lambda x, y: pd.merge(x, y, how='outer', 
                                            left_index=True, right_index=True),
    [vol_daily1, vol_daily2, vol_daily3, vol_daily4, vol_daily5, vol_daily6])

df_vol_daily.to_csv(out_dir + 'daily_vol_pred_results2_norm.csv')


# =============================================================================
# 3 month weekly
# =============================================================================

cols_vol4 = ['asvi' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol5 = ['atap' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol6 = (cols_vol4 + cols_vol5)[:-1]
X_asvi = sm.add_constant(df13_weekly_vol_pred.loc[:, cols_vol4])
X_atap = sm.add_constant(df13_weekly_vol_pred.loc[:, cols_vol5])
X_alll = sm.add_constant(df13_weekly_vol_pred.loc[:, cols_vol6])
Y = df13_weekly_vol_pred.vol0

ols_weekly1 = [get_ols_result(get_vol_pred(Y, X)) for X in [X_asvi, X_atap, X_alll]]

# =============================================================================
# 10 year weekly
# =============================================================================

cols_vol4 = ['asvi' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol5 = ['atap' + str(i) for i in np.arange(6)] + ['vol1']
cols_vol6 = (cols_vol4 + cols_vol5)[:-1]
X_asvi = sm.add_constant(df10_weekly_vol_pred.loc[:, cols_vol4])
X_atap = sm.add_constant(df10_weekly_vol_pred.loc[:, cols_vol5])
X_alll = sm.add_constant(df10_weekly_vol_pred.loc[:, cols_vol6])
Y = df10_weekly_vol_pred.vol0
ols_weekly2 = [get_ols_result(get_vol_pred(Y, X)) for X in [X_asvi, X_atap, X_alll]]

df_vol_weekly = reduce(lambda x, y: pd.merge(x, y, how='outer',
                                             left_index=True, right_index=True),
                       ols_weekly1 + ols_weekly2)

df_vol_weekly.to_csv(out_dir + 'weekly_vol_pred_results2_norm.csv')

# =============================================================================
# Granger causality test
# =============================================================================
grangercausalitytests(df_pred.loc[:, ['atap0', 'asvi0']], 6)
grangercausalitytests(df_pred.loc[:, [ 'asvi0', 'atap0']], 6)

# =============================================================================
# Impulse response - daily
# =============================================================================

results = VAR(df_pred.loc[:, ['atap0', 'asvi0', 'ret0']]
                     .rename(columns={'atap0': 'ATAP', 
                                      'asvi0': 'ASVI',
                                      'ret0': 'Return'})).fit(6)
# results.plot()
# results.plot_acorr()

irf = results.irf(10)
fig = irf.plot(orth=True)
fig.savefig(out_dir + 'daily13_impulse.png', dpi=400)

results10 = VAR(df_pred2.loc[:, ['atap0', 'asvi0', 'ret0']]
                        .rename(columns={'atap0': 'ATAP', 
                                      'asvi0': 'ASVI',
                                      'ret0': 'Return'})).fit(6)

irf10 = results10.irf(10)
fig2 = irf10.plot(orth=True)
fig2.savefig(out_dir + 'daily10_impulse.png', dpi=400)


# =============================================================================
# Impulse response - weekly
# 
# =============================================================================
results_week = VAR(df_pred_weekly.loc[:, ['atap0', 'asvi0', 'ret0']]
                     .rename(columns={'atap0': 'ATAP', 
                                      'asvi0': 'ASVI',
                                      'ret0': 'Return'})).fit(6)
# results.plot()
# results.plot_acorr()

irf_week = results_week.irf(10)
fig_week = irf_week.plot(orth=True)
fig_week.savefig(out_dir + 'weekly13_impulse.png', dpi=400)

results10_week = VAR(df_pred_weekly2.loc[:, ['atap0', 'asvi0', 'ret0']]
                        .rename(columns={'atap0': 'ATAP', 
                                      'asvi0': 'ASVI',
                                      'ret0': 'Return'})).fit(6)

irf10_week = results10_week.irf(10)
fig2_week = irf10_week.plot(orth=True)
fig2_week.savefig(out_dir + 'weekly10_impulse.png', dpi=400)
