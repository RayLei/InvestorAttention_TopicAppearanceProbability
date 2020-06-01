#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:53:00 2020

The script plot the accumulated daily Topic Appearance Probability of
'Monetary Policy' against the date of FOMC meetings.

@author: LEIHAO
"""

from os import listdir
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import matplotlib.lines as mlines



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

# =============================================================================
# Plot Fed topic
# =============================================================================
fed_reg_st_date = [
        date(2012, 7, 31),  date(2012, 10, 23),
        date(2013, 1, 29),  date(2013, 4, 30),
        date(2013, 7, 30),
        date(2013, 10, 29),
        date(2014, 1, 28),  date(2014, 4, 29),
        date(2014, 7, 29),
        date(2014, 10, 28),
        date(2015, 1, 27),  date(2015, 4, 28),
        date(2015, 7, 28),
        date(2015, 10, 27),
        date(2016, 1, 26),  date(2016, 4, 26),
        date(2016, 7, 26),
        date(2016, 11, 1),
        date(2017, 1, 31), date(2017, 3, 14)
        ]
fed_sig_date = [date(2012, 3, 13), date(2013, 10, 16), date(2014, 3, 4)]

fed_anc_date = [
        date(2012, 1, 24), date(2012, 4, 24), date(2012, 6, 19),
        date(2012, 9, 12),
        date(2012, 12, 11),
        date(2013, 3, 19), date(2013, 6, 18), date(2013, 9, 17),
        date(2013, 12, 17),
        date(2014, 3, 18), date(2014, 6, 17), date(2014, 9, 16),
        date(2014, 12, 16),
        date(2015, 3, 17), date(2015, 6, 16), date(2015, 9, 16),
        date(2015, 12, 15),
        date(2016, 3, 15), date(2016, 6, 14), date(2016, 9, 20),
        date(2016, 12, 13),
        ]

jacksonhole_date = [
        date(2012, 8, 30),
        date(2013, 8, 21),
        date(2014, 8, 21),
        date(2015, 8, 27),
        date(2016, 8, 26),
        ]
colour = 'orange'
alpha = 0.7
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pd.to_datetime(df_tap.index), df_tap.values,
        color='black', alpha=1)
for st_date in fed_reg_st_date:
    p1 = ax.axvline(x=st_date, color=colour, alpha=alpha, ls=':', label='schedualed')
    # ax.axvline(x=st_date + timedelta(days=1), color='red', alpha=alpha, ls=':')
    # ax.axvspan(st_date, st_date + timedelta(days=1), color='red', alpha=0.2)

for sig_date in fed_sig_date:
    p2 = ax.axvline(x=sig_date, color=colour, alpha=alpha, ls='-.', label='unscheduled')

for pres_date in fed_anc_date:
    ax.axvline(x=pres_date, color=colour, alpha=alpha, ls='--')

#for jack_date in jacksonhole_date:
#    ax.axvline(x=jack_date, color=colour, alpha=0.2, ls='-')

#ax.text(x=date(2013, 10, 9), y=11, 'Yellen nomination date')
ax.annotate('Yellen nomination day', xy=(date(2013, 10, 9), 11.278),
            xytext=(date(2013, 11, 30), 12.),
            arrowprops=dict(facecolor='black', arrowstyle='->'))


ax.annotate('Jackson Hole', xy=(date(2016, 8, 26), 4.34),
            xytext=(date(2016, 9, 26), 6),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('Jackson Hole', xy=(date(2014, 8, 21), 3.9),
            xytext=(date(2014, 9, 22), 6),
            arrowprops=dict(facecolor='black', arrowstyle='->'))
ax.annotate('Yellen speech', xy=(date(2016, 3, 29), 3.5),
            xytext=(date(2015, 10, 10), 4.5),
            arrowprops=dict(facecolor='black', arrowstyle='->'))


dotline = mlines.Line2D([], [], linestyle=':', color=colour, alpha=0.7)
dadline = mlines.Line2D([], [], linestyle='-.', color=colour, alpha=0.7)
solline = mlines.Line2D([], [], linestyle='-', color='k', alpha=0.7)
dasline = mlines.Line2D([], [], linestyle='--', color=colour, alpha=0.7)

ax.legend((dotline, dadline, dasline, solline),
          ('schedualed', 'unschedualed', 'with press conference', 'topic probability'),
          loc='upper right', prop={'size': 8}, shadow=False)
fig.tight_layout()
out_dir = '/Users/leihao/Documents/ATAP/'

fig.savefig(out_dir + 'fed_tap_fomc.png', dpi=400)
# =============================================================================
# investigate 2016 peaks
# =============================================================================
df_mtne_2016 = df_mtne_tpc2[df_mtne_tpc2.year==2016]
df_mtne_2016.nlargest(10, 3)
