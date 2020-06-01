#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:17:17 2020

The script converts table to the form used in latex table. 

@author: LEIHAO
"""

from os import listdir
import pandas as pd
import re


in_dir = '/Users/leihao/Documents/ATAP/'

files = [f for f in sorted(listdir(in_dir))
            if 'p_treasury_ret_pred_daily_results_final_norm_win' in f
            or 'p_treasury_ret_pred_week_results_final_norm_win' in f]
files_csv = [f for f in files if 'csv' in f]


def add_star(row):
    x, p = row
    if not row.name.startswith('adj_r2'):
        if p < 0.01:
            x += '***'
        elif p < 0.05:
            x += '**'
        elif p < 0.1:
            x += '*'
    return x


def modify_coef(df, ap):
    df.loc[:, 'coef_' + ap] = df.loc[:, ['coef_' + ap, 'P>|t|_' + ap]].apply(add_star, axis=1)
    return df


def concat_coef(df, ap):
    s1 = df.loc[:, 'coef_' + ap]
    s2 = df.loc[:, 'std err_' + ap].apply(lambda x: '(' + x + ')')
    s1.index = s1.index.map(lambda x: x + '_c')
    s2.index = s2.index.map(lambda x: x + '_s')
    return(pd.concat((s1, s2), axis=0))


def get_latex_table(f, appd=['x_x', 'y_x', 'x', 'x_y', 'y_y', 'y']):
    df = pd.read_csv(in_dir + f, index_col=0)
    coef_cols = [c for c in df.columns if 'coef' in c]
    df.loc[:, coef_cols] = df.loc[:, coef_cols]\
                             .applymap(lambda x: "{:.3f}".format(x))
    std_cols = [c for c in df.columns if 'std err' in c]
    df.loc[:, std_cols] = df.loc[:, std_cols]\
                            .applymap(lambda x: "{:.2f}".format(x))

    # appd = ['x_x', 'y_x', 'x', 'x_y', 'y_y', 'y']
    for ap in appd:
        df = modify_coef(df, ap)

    df_tar = pd.concat([concat_coef(df, ap) for ap in appd], axis=1).sort_index()
    df_tar = df_tar.drop(index='adj_r2_s')
    return df_tar


def get_name(f):
    grp = re.search(r'p_treasury_ret_pred_(.*)_results_final(.*).csv', f)
    name = grp.group(1) + grp.group(2)
    return name


names = [get_name(f) for f in files_csv]
all_tables = [get_latex_table(f) for f in files_csv]
xls = pd.ExcelWriter(in_dir + 'p_ret_pred_4latex_norm.xlsx')
for name, table in zip(names, all_tables):
    table.astype('str').to_excel(xls, name)
xls.save()

# =============================================================================
# 3-month no outlier
# =============================================================================
filenames = [f for f in sorted(listdir(in_dir)) if 'p_3m_nooutlier' in f]
appd = ['x', 'y', 'z']


def get_name(f):
    grp = re.search(r'p_3m_nooutlier_ret_pred_(.*)_results_norm_win(.*).csv', f)
    name = grp.group(1) + grp.group(2)
    return name


names = [get_name(f) for f in filenames]
all_tables = [get_latex_table(f, appd) for f in filenames]
xls = pd.ExcelWriter(in_dir + 'p_3m_nooutlier_ret_pred_4latex_norm.xlsx')
for name, table in zip(names, all_tables):
    table.astype('str').to_excel(xls, name)
xls.save()

# =============================================================================
# contemporaneous relation
# =============================================================================
filenames = ['p_treasury_ret_exp_results_norm.csv',
             'p_treasury_ret_exp_week_results_norm.csv']
names = ['daily', 'weekly']
all_tables = [get_latex_table(f) for f in filenames]
xls = pd.ExcelWriter(in_dir + 'p_ret_exp_4latex_norm.xlsx')
for name, table in zip(names, all_tables):
    table.astype('str').to_excel(xls, name)
xls.save()

# =============================================================================
# monthly volatility
# =============================================================================
filename = 'monthly_vol_pred_results_win5_norm.csv'
table = get_latex_table(filename)
table.astype('str').to_excel(in_dir + 'p_monthly_vol_norm_4latex.xlsx')

# =============================================================================
# daily varaince
# =============================================================================
filename = 'daily_vol_pred_results2_norm.csv'
table = get_latex_table(filename)
table.astype('str').to_excel(in_dir + 'p_daily_vol_norm_4latex.xlsx')

# =============================================================================
# weekly varaince
# =============================================================================
filename = 'weekly_vol_pred_results2_norm.csv'
table = get_latex_table(filename)
table.astype('str').to_excel(in_dir + 'p_weekly_vol_norm_4latex.xlsx')
