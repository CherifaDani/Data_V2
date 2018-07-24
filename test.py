# -*- coding: utf-8 -*-
from Variable import Variable
import derivation_calculs
import derivation_functions
import compare_dfs
import pandas as pd

script_path = '1V.xlsx'
state_path = 'variable_state.csv'
# var_name = 'STR_USD_1D_Z250D'  # ewma
# var_name = 'STR_USD_1D'
# var_name = 'FUT_BUND_TREND6M'  # ewma
# var_name = 'STR_USD_1D_DACE_1_20_100'  #  autocorr
# var_name = 'SWAP_INFLA_EUR_2Y_DACE_1_20_100'
# var_name = 'STR_USD_1D_1D_STD20'  # vol
# var_name = 'FUT_CL_RET1_STD50'  # vol
# var_name = 'EURJPY_FWDRET20'  # PCT_DELTA
# var_name = 'FUT_ESTX50_C1_RET1D'  # rolling_return, csv absent
# var_name = 'SP500_RET1'  # rolling_return
# var_name = 'Spread_GOV_USA_30Y-GOV_JPN_30Y'  # combi
# var_name = 'Spread_SWAP_USD_2Y-GOV_USA_2Y'  # combi
# var_name = 'CORP_IG_USA_5Y'  # combi
# var_name = 'USDEUR_CR20'  # combi
# var_name = 'SP500_EBIT1Y_YIELD'  # combi
# var_name = 'SP500_DPS_YIELD'  # combi
# var_name = 'SP500_CR20'  # combi
# var_name = 'SP500_RCR20'  # csv absent!
# var_name = 'SP500_RCR20_SGN'  # var not found
# var_name = 'Corr_GOV_USA_10Y-GOV_GER_10Y'  # corr
# var_name = 'Corr_BOVES_LAST-Gold'  # corr
# var_name = 'Corr_WTI-Gold'  # corr
# var_name = 'FUT_CL_NAV'  # cumret
var_name = 'Gold_DACE_1_20_100'

# var_name = 'FUT_ESTX50_RET1ROLL'  # futures_roll csv absent
# var_name = 'FUT_JNI_RET1ROLL'  # futures_roll
# var_name = 'SP500_RET1'  # rolling_return
# var_name = 'STR_JPY_1D_DACE_1_20_100'  # autocorr
# var_name = 'CDS_UK_1Y_DACE_1_20_100'  # autocorr
# var_name = 'GOV_USA_1Y_DACE_1_20_100'  # autocorr
# var_name = 'FRA_USD_6X12_DACE_1_20_100'

#
b = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name)

b.write_dict()
var_name2 = b.get_param('parents')
operation = (b.get_param('operation'))
parameters = b.get_param('derived_params')
dfx = b.read_var(b.get_param('path'))

c = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name2[0])
c.write_dict()
# d = Variable(script_path=script_path,
#              state_path=state_path,
#              var_name=var_name2[1])
# d.write_dict()
df = derivation_calculs.apply_operation(var_list=[c], freq='B', operation=operation, parameters=parameters)
print '################################DF##########################################'
print(df)
df.to_csv('x_test.csv')
print compare_dfs.compare_two_dfs(dfx, df)





















# if len(var_name2) > 1:
#     d = Variable(script_path=script_path,
#                  state_path=state_path,
#                  var_name=var_name2[1])
#     d.write_dict()
#     df = derivation_calculs.apply_operation(var_list=[c, d], freq='B', operation=operation, parameters=parameters)
#
#
# else:
#     df = derivation_calculs.apply_operation(var_list=[c], freq='B', operation=operation, parameters=parameters)
# print(df)
# dfy = c.read_var(c.get_param('path'))
# print(dfy)
# emadecay = parameters['emadecay']
# span = 2.0 / (emadecay - 1)
# dfs = pd.ewma(dfy, span=span)
# dfc = c.read_var(c.get_param('path'))
# print(b.get_param('parents'))
# # print b.update()
# print 'dfx: {}'.format(dfx)

# print derivation_functions.apply_lag(dfx, lag=1, freq='B', cols=None, inplace=False)

#
# cols1 = get_columns(df1)
# cols2 = get_columns(df2)
# cols = [cols1, cols2]
# if cols is None:
#     return None
# # dfx, dfy = reindex(df1, df2)
# # print(dfx.shape)
# df = pd.concat([df1, df2], axis=1)
# # colx = df.columns
# # cols = get_columns(df)
# maincol = cols[0]
# substcol = cols[1]
# datacols = df[maincol]
# print(datacols)
# # maincol = df1
# # substcol = df2
# # datacols = df1
# if not inpct:
#     retdata = df.diff(period, axis=1)
# else:
#     retdata = df.pct_change(period, axis=1)
#
# idx_all = pd.bdate_range(start=df.index[0], end=df.index[-1], freq='B')
# # effacer l'heure pour synchronisation
# retdata = set_daytime(retdata, datetime(2000, 1, 1))
# df = set_daytime(df, datetime(2000, 1, 1))
# # élargir le calendrier pour inclure les dates de rolls de façon certaine
# retdata = retdata.reindex(index=idx_all, method=None)
#
# # générer la série des dates de roll
# #          if rollfreq [1:].find('BS') < 0:
# #              rollfreq=rollfreq + 'BS'
# rolldates = pd.bdate_range(start=df.index[0], end=df.index[-1], freq=rollfreq)
# rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)
# # Ne garder que les dates de roll antérieures aux données courantes
# rolldates = rolldates[rolldates <= retdata.index[-1]]
# daybefore_rolldates = rolldates + pd.datetools.BDay(-period)
# dayafter_rolldates = rolldates + pd.datetools.BDay(period)

# timeidx=self.index
# Contrat M(front) coté jusqu'à TRoll, traité jusqu'en TRoll-1, roulé en TRoll-1
# Contrat M+1(next), coté jusqu'à TRoll, devient le front en TRoll + 1, traité en TRoll-1
# Returns:
#  en TRoll, Close(F2, TRoll)/ Close(F2, TRoll-1) - 1
#  en TRoll + 1, Close(F1, TRoll+1)/ Close(F2, TRoll-1) - 1
# dayafter_roll_contract=maincol
# cas de UX
# if effectiveroll_lag == 0:
#     roll_contract = maincol
#     daybefore_roll_contract = maincol
# # cas de FVS
# else:
#     roll_contract = substcol
#     daybefore_roll_contract = substcol
#
# if inpct:
#     rollday_returns = df.loc[rolldates, roll_contract].values / \
#                       df.loc[daybefore_rolldates, daybefore_roll_contract].values - 1
#     dayafter_returns = df.loc[dayafter_rolldates, maincol].values / \
#                        df.loc[rolldates, substcol].values - 1
# else:
#     rollday_returns = df.loc[rolldates, roll_contract].values - \
#                       df.loc[daybefore_rolldates, daybefore_roll_contract].values
#     dayafter_returns = df.loc[dayafter_rolldates, maincol].values - \
#                        df.loc[rolldates, substcol].values
# retdata.loc[rolldates, maincol] = rollday_returns
# retdata.loc[dayafter_rolldates, maincol] = dayafter_returns
# newdataset = pd.DataFrame(index=df.index,
#                           data=df.values,
#                           columns=['ROLLING_RETURN'])
# # revenir au calendrier restreint
# newdataset = pd.DataFrame(newdataset.dropna())
# return newdataset
