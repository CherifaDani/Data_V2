# -*- coding: utf-8 -*-
from Variable import Variable
import derivation_calculs
import derivation_functions
import compare_dfs
import pandas as pd
import data_utils
import numpy as np
import time

script_path = '1v.xlsx'
state_path = 'variable_state.csv'
"""
Date: 30-07-2018

# var_name = 'FUT_CL_NAV'  # cumret (ok)
# var_name = 'STR_USD_1D'  # ok
# var_name = 'STR_USD_1D_1D_STD20'  # vol  ok
# var_name = 'FUT_CL_RET1_STD50'  # vol ok
# var_name = 'FUT_BUND_TREND6M'  # ewma ok
# var_name = 'SWAP_INFLA_EUR_2Y_DACE_1_20_100'  # 20 differences [Dates au lieu de NaN]
# var_name = 'GOV_USA_1Y_DACE_1_20_100'  # autocorr [3 differences]; [Dates au lieu de NaN]
# var_name = 'SP500_RET1'  # rolling_return ok
# var_name = 'STR_JPY_1D_DACE_1_20_100'  # autocorr  78 diff
# var_name = 'FUT_ESTX50_RET1ROLL'  # futures_roll ok
# var_name = 'GOV_GER_2Y_DACE_1_20_100'  # ok [ 11-09-2001]  params:
              # {'period': 1,'shortwindow': 20, 'longwindow' :100, 'inpct' : False,
              # 'exponential' : False, 'lag': 1, 'col_out':[0]}
# var_name = 'SP500_DY 12M_DACE_1_20_100'  # 113 diff
var_name = 'FUT_BUND_SGN_CR5'  # combi ok [lin: True, Transfo: sign]
var_name = 'FUT_BUND_DRIFT'  # combi 2 diff [lin:True}]
var_name = 'Gold_DACE_1_20_100'  # ok
var_name = 'SP500_RET1'  # rolling_return ok
var_name = 'GOV_USA_1Y_DACE_1_20_100'  # autocorr

var_name = 'FUT_BUND_FWDRET5'  # pctdelta [97 diff]
# var_name = 'FUT_BUND_TREND6M'  # ewma ok
var_name = 'STR_USD_6M_Z5D'  # ewma ok
var_name = 'FUT_BUND_CR1'  # combi ok
var_name = 'Corr_GOV_USA_10Y-GOV_GER_10Y'
var_name = 'GOV_GER_2Y_DACE_1_20_100' ok
var_name = 'Corr_GOV_GER_10Y-GOV_JPN_10Y'  #  ok
var_name = 'Spread_SWAP_USD_2Y-GOV_USA_2Y'  # combi
var_name = 'SP500_EBIT1Y_YIELD'  # combi
var_name = 'CORP_IG_USA_5Y'  # combi
var_name = 'Corr_BOVES_LAST-Gold'  # corr
var_name = 'Corr_GOV_USA_10Y-GOV_GER_10Y'  # corr
# var_name = 'GOV_GER_2Y_DACE_1_20_100'

# var_name = 'FUT_BUND_RET1C1' ok
# var_name = 'Corr_BOVES_LAST-Gold'  # corr
# var_name = 'Corr_TOPIX_LAST-HSCE_LAST'
# var_name = 'Corr_HSCE_LAST-NSDQ_LAST' ok
# var_name = 'Corr_NIKKEI225_LAST-TOPIX_LAST' ok
# var_name = 'FUT_TNOTE_RET1C1' # pct_delta ok
# var_name = 'Spread_GOV_USA_30Y-GOV_JPN_30Y'  # combi  ok

# BaD

# var_name = 'Corr_WTI-Gold'  # corr {'period':1 , 'window': 100, 'inpct': True, 'exponential':True, 'col1':0, 'col2': 1}



# calcul incrémental
var_name = 'CORP_IG_USA_5Y_Z45D'  # ewma, incr done!
var_name = 'FUT_BUND_DRIFT'  # combi incr ok!

# var_name = 'Spread_GOV_USA_2Y-GOV_USA_1Y'
# var_name = 'Spread_GOV_USA_30Y-GOV_GER_30Y'
# var_name = 'CORP_HY_US_5Y'
var_name = 'Spread_GOV_USA_30Y-GOV_JPN_30Y'  # combi not ok



# PBs
# var_name = 'TOPIX_FIN_DY 12M_DACE_1_20_100'  # csv absent
# var_name = 'CADUSD_Open_DACE_1_20_100'  # csv absent
# var_name = 'EURJPY_FWDRET20'  # PCT_DELTA not found
# var_name = 'STR_USD_1D_Z250D'  # ewma, csv absent
# var_name = 'FUT_ESTX50_C1_RET1D'  # rolling_return, csv absent
# var_name = 'SP500_RCR20'  # csv absent!
var_name = 'SP500_RCR20_SGN'  # var not found
var_name = 'FUT_BUND_SGN_CR20'  # bad line
var_name = 'FRA_USD_6X12_DACE_1_20_100'  # var not found
var_name = 'CDS_UK_1Y_DACE_1_20_100'  # autocorr  path error
# var_name = 'FUT_BUND_RCR20' files not found
var_name = 'FUT_JNI_RET1ROLL'  # futures_roll var not found
var_name = 'SP500_DPS_YIELD'  # path error
var_name = 'SP500_DPS_YIELD'  # combi
var_name = 'USDEUR_CR20'  # combi
# var_name = 'FUT_GBPUSD_DRIFT'  # no path
var_name = 'FUT_JGB_CR20'
var_name = 'FUT_SP500_C1_RET1D'
# var_name = 'FUT_CL_C1_RET1D'
var_name = 'FUT_XB_C1_RET1D'
# var_name = 'FUT_NKY_C1_RET1D'

"""
var_name = 'Spread_GOV_USA_2Y-GOV_USA_1Y'
# var_name = 'Corr_WTI-Gold'

# var_name = 'FUT_BUND_TREND6M'  # ewma # var_name = 'FUT_BUND_TREND6M'  # ewma ok
b = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name)
print b.update()

# Dérivation partielle
# start = time.time()
# b = Variable(script_path=script_path,
#              state_path=state_path,
#              var_name=var_name)
#
# b.write_dict()
# print b.get_params()
#
# var_name2 = b.get_param('parents')
# print('parents {}'.format(var_name2))
#
# operation = b.get_param('operation')
# parameters = b.get_param('derived_params')
#
# df_derived = b.read_var()
#
# c = Variable(script_path=script_path,
#              state_path=state_path,
#              var_name=var_name2[0])
#
# c.write_dict()
# print('path_parents {}'.format(c.get_param('path')))
#
#
# if len(var_name2) > 1:
#     d = Variable(script_path=script_path,
#                  state_path=state_path,
#                  var_name=var_name2[1])
#     d.write_dict()
#     df = derivation_calculs.apply_operation(var_list=[c, d], freq='B', operation=operation,
#                                             parameters=parameters)
#
# else:
#     df = derivation_calculs.apply_operation(var_list=[c], freq='B', operation=operation,
#                                             parameters=parameters)
#
# print '################################DF##########################################'
# print('df calculé {}'.format(df))
# df.to_csv('x_test.csv')
# dfx = b.read_var(b.get_param('path'))
# print('df déjà présent {}'.format(dfx))
# print compare_dfs.compare_two_dfs(dfx, df)
