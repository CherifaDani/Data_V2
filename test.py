# -*- coding: utf-8 -*-
from Variable import Variable
import derivation_calculs
import derivation_functions
import compare_dfs
import pandas as pd
import data_utils
import numpy as np
import time

script_path = '1V.xlsx'
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




# BaD
# var_name = 'Corr_GOV_GER_10Y-GOV_JPN_10Y'  # not ok
# var_name = 'Corr_WTI-Gold'  # corr {'period':1 , 'window': 100, 'inpct': True, 'exponential':True, 'col1':0, 'col2': 1}





# var_name = 'TOPIX_FIN_DY 12M_DACE_1_20_100'  # csv absent
# var_name = 'CADUSD_Open_DACE_1_20_100'  # csv absent
# var_name = 'EURJPY_FWDRET20'  # PCT_DELTA not found
# var_name = 'STR_USD_1D_Z250D'  # ewma, csv absent
# var_name = 'FUT_ESTX50_C1_RET1D'  # rolling_return, csv absent
# var_name = 'SP500_RCR20'  # csv absent!
var_name = 'SP500_RCR20_SGN'  # var not found
var_name = 'FUT_BUND_SGN_CR20'  # bad line

"""

var_name = 'Spread_GOV_USA_30Y-GOV_JPN_30Y'  # combi not ok
# var_name = 'Spread_SWAP_USD_2Y-GOV_USA_2Y'  # combi not ok
# var_name = 'CORP_IG_USA_5Y'  # combi
# var_name = 'USDEUR_CR20'  # combi
# var_name = 'SP500_EBIT1Y_YIELD'  # combi
# var_name = 'SP500_DPS_YIELD'  # combi
# var_name = 'SP500_CR20'  # combi
# var_name = 'Corr_GOV_USA_10Y-GOV_GER_10Y'  # corr
# var_name = 'Corr_BOVES_LAST-Gold'  # corr
# var_name = 'Gold_DACE_1_20_100'
# var_name = 'CORP_IG_USA_5Y'
# var_name = 'FUT_BUND_RCR20'
# var_name = 'FUT_BUND_CR1'  # combi
# var_name = 'FUT_JNI_RET1ROLL'  # futures_roll
# var_name = 'SP500_RET1'  # rolling_return
# var_name = 'CDS_UK_1Y_DACE_1_20_100'  # autocorr
# var_name = 'GOV_USA_1Y_DACE_1_20_100'  # autocorr
# var_name = 'FRA_USD_6X12_DACE_1_20_100'
start = time.time()
b = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name)
# b.update()
# print(pd.read_csv('2 Data/2 Calculs/18 04 Derived Lab/X/STR_USD_1D_Z250D.csv'))
# print(time.time() - start)
b.write_dict()
var_name2 = b.get_param('parents')
operation = b.get_param('operation')
parameters = b.get_param('derived_params')

c = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name2[0])
print(var_name2)
c.write_dict()
# # # #
# d = Variable(script_path=script_path,
#              state_path=state_path,
#              var_name=var_name2[1])
# d.write_dict()


if len(var_name2) > 1:
    d = Variable(script_path=script_path,
                 state_path=state_path,
                 var_name=var_name2[1])
    d.write_dict()
    df = derivation_calculs.apply_operation(var_list=[c, d], freq='B', operation=operation, parameters=parameters)

else:
    df = derivation_calculs.apply_operation(var_list=[c], freq='B', operation=operation, parameters=parameters)
print '################################DF##########################################'
# print(df)
# df.to_csv('x_test.csv')
# dfx = b.read_var(b.get_param('path'))
# print compare_dfs.compare_two_dfs(dfx, df)
