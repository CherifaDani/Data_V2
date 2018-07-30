# -*- coding: utf-8 -*-
# @PydevCodeAnalysisIgnore
from __future__ import division
from Variable import Variable
import derivation_functions

from datetime import datetime
import pandas as pd
import control_utils as cu
import csv
import data_utils
from os.path import basename
import os
import data_utils
import sys
import os
import sys
import glob
import warnings
import csv
import numpy as np
import pandas as pd
from datetime import date
from os.path import join, splitext, basename, dirname
import zipfile
import ast
from xlrd import XLRDError
import derivation_calculs
from compare_dfs import compare_two_dfs
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
# graphviz = GraphvizOutput()
# graphviz.output_file = 'basic.png'
# # import derivation_calculs
# sys.path.append(r'/home/cluster')
# with PyCallGraph(output=graphviz):
# script_path = 'CM.xlsx'
# var_name = 'Corr_VIX_LAST-WTI'
# var_name = 'Corr_GSCI-Gold'
# var_name = 'STR_USD_1D'
# var_name = 'Spread_CDS_CHN_5Y-CDS_USA_5Y'
# var_name = 'STR_USD_1D_DACE_1_20_100'
# state_path = 'x.csv'
# state_path = 'variable_state.csv'
# var_name = 'FUT_EURUSD_RET1C1'
# var_name = 'STR_USD_1M'
# var_name = 'Spread_SWAP_USD_1Y-GOV_USA_1Y'
# var_name = 'FUT_BUND_RET1C1'
# var_name = 'SP500_TREND6M'  # to test before very complex variable
# var_name = 'FUT_BUND_TREND6M'  # complex variable ==> futures_roll
# var_name = 'CDS_GER_1Y_Z250D'
# var_name = 'CDS_GER_1Y'
# var_name = 'STR_USD_1D_Z5D'
#
df1 = data_utils.load_var('compare/1.csv', '1')
df2 = data_utils.load_var('compare/2.csv', '2')
print compare_two_dfs(df1, df2)
# # df.set_index('')
# print(df1.shape, df2.shape)
# df3 = pd.concat([df1, df2], axis=1)
# print(df3.shape)
# x = df1.columns[1]
# print(x)
# print(df1[x])

# col_out = [0]
# output_df = df1[df1.columns[col_out]]
# print(output_df)











# var_name = 'CREDIT_CDS_GER_1Y'
# var_name2 = 'CDS_GER_1Y_1D_STD20'
# var_name = 'CDS_GER_1Y_DACE_1_20_100'
# var_name = 'HSCE_CLOSE_Z5D'


# var_name = 'GOV_USA_10Y_DACE_1_20_100'
#
# c = Variable(script_path=script_path,
#              state_path=state_path,
# #              var_name=var_name2)
# b = Variable(script_path=script_path,
#              state_path=state_path,
#              var_name=var_name)
# # path = '2 Data/1 Received/Market data/Base/CREDIT_CDS_GER_1Y.csv'
# # print(pd.read_csv(path))
# print b.update()
# c.write_dict()
# b.write_dict()
#
# df = data_utils.load_var(b.get_param('path'), 'credit_cds')
# df2 = data_utils.load_var('2.csv', '1')
# todate = pd.to_datetime('08-05-2018', dayfirst=True)
# print(df)
# print(derivation_functions.extendtodate(df, todate=todate, freq='B', limit=5))
# df2 = c.read_var(c.get_param('path'))
# print(df2)
# df1 = derivation_functions.apply_timeshift(df, var_name=b.get_param('var_name'), shift=1, freq='B', ownfreq=None, refdate=None)
# print b.update()
# df1 = data_utils.load_var(c.get_param('path'), 'credit_cds')




# print derivation_functions.get_columns(df, cols=None, forceuppercase=True)
# print derivation_functions.apply_combi(df, df, var_name='ff', idx1=0, coeff1=1, idx2=1, coeff2=5, constant=0,
#                 inplace=False, islinear=True, transfo=None)

#
# print derivation_functions.apply_ewma(df, var_name='ewma', emadecay=None, span=1, inplace=True,
#                cols=None, wres=True, normalize=False,
#                histoemadata=None, overridedepth=0,
#                fieldsep='', stdev_min=1e-5)

# print derivation_functions.apply_corr(df, df2, 'ff', period=0,
#               span=20, exponential=True,
#               inpct=True, cols='COMMO_BRENT', lag2=0)

# print derivation_functions.take_diff(df, var_name='ff', period=1, inplace=False, cols=None, inpct=True,
#               fieldsep='', alldays=True, ownfreq=None)
# print derivation_functions.estimate_nat_freq(df, 'COMMO_BRENT')
# cols = df.columns.values
# cols = data_utils.check_cell(cols)
# print derivation_functions.apply_vol(df, 'ff')
# print df2

# print derivation_functions.set_daytime(df, datetime(2018,07,13,16,25,47,252))
# print df2.index.values
# print derivation_functions.rolling_returns(df,
#                                             df,
#                                             var_name='vv',
#                                             rollfreq='B',
#                                             iday=1,
#                                             iweek=1,
#                                             effectiveroll_lag=0,
#                                             inpct=True)















# df1 = data_utils.load_var('compare/1.csv', '1')
# df2 = data_utils.load_var('compare/2.csv', '2')
# # # df1 = pd.read_csv('compare/2.csv')
# # # print(df1)
# print compare_two_dfs(df1, df2)
# print data_utils.alterfreq(path=path, freq='B', refdate='', last_update=last_update, var_name=b.get_param('var_name'))
# print type(refdate) in [datetime]
# b.write_dict()

# print b.write_dict()
# c.write_dict()
# # #print b.read_var(b.get_param('path'))
# vlist = [b, b]

# parameters = {'period' : 1, 'freq' : 'BQ', 'week': 1, 'weekday': 2, 'day': 25, 'col1': 0, 'col2': 1,
#               'col_out': 'RETURN_1D_AFTER_ROLL', 'bday_offset': -1,
# #               'bmonth_offset': -1}
# operation = 'vol'
# print derivation_calculs.apply_operation(vlist, 'B', operation, b.get_param('derived_params'))

# print b.write_dict()
# print(b.test())












# df = data_utils.load_var(''2 Data/2 Calculs/18 05 derived data/X/CDS_GER_1Y_DACE_1_20_100.csv', '1')

# stdev_min = 1e-5
# # calculer la période synthétique correspondant au coeff s'il est fourni
# emadecay = 2/(1+45)
# wres = True
# wz = True
# if type(emadecay) in [int, float]:
#     if emadecay > 0:
#         span = (2.0 / emadecay) - 1
# df_calc = pd.ewma(df, span=span, adjust=True)
# if wres:
#         rescols = df - df_calc
#         # calcul du ZScore
#         if wz:
#             stdevcols = pd.ewmstd(rescols, span=span)
#             stdevcols[stdevcols <= stdev_min] = np.nan
#             zcols = rescols * 0.0
#             zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]
# print zcols[stdevcols > 0]
# # calcul du résidu
# if wres:
#     rescols = df - df_calc
#     # calcul du ZScore
#     if wz:
#         stdevcols = pd.ewmstd(rescols, span=span)
#         stdevcols[stdevcols <= stdev_min] = np.nan
#         zcols = rescols * 0.0
#         zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]
