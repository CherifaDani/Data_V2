# -*- coding: utf-8 -*-
# @PydevCodeAnalysisIgnore
from __future__ import division
from Variable import Variable
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
from os.path import join, splitext, basename, dirname
import zipfile
import ast
from xlrd import XLRDError
import derivation_calculs
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
graphviz = GraphvizOutput()
graphviz.output_file = 'basic.png'
#import derivation_calculs
#sys.path.append(r'/home/cluster')
with PyCallGraph(output=graphviz):
    script_path = 'deriv_script.xlsx'
    #var_name = 'Corr_VIX_LAST-WTI'
    #var_name = 'Corr_GSCI-Gold'
    var_name = 'STR_USD_1D'
    #var_name2 = 'Spread_SWAP_USD_1Y-GOV_USA_1Y'
    state_path = 'variable_state.csv'
    #var_name = 'FUT_EURUSD_RET1C1'
    b = Variable(script_path=script_path,
                 state_path=state_path,
                 var_name=var_name)
    # c = Variable(script_path=script_path,
    #               state_path=state_path,
    #               var_name=var_name2)
    print b.update
# b.write_dict()
#===============================================================================
# b.write_dict()
# # c.write_dict()
# #print b.read_var(b.get_param('path'))
# vlist = [b]
# parameters = {'period': 1}
# operation = 'pctdelta'
# print derivation_calculs.apply_operation(vlist, 'B', operation, parameters)
# 
#===============================================================================
# df = data_utils.load_var('1.csv', '1')
# #===============================================================================
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
#===============================================================================
#===============================================================================
# # calcul du résidu
# if wres:
#     rescols = df - df_calc
#     # calcul du ZScore
#     if wz:
#         stdevcols = pd.ewmstd(rescols, span=span)
#         stdevcols[stdevcols <= stdev_min] = np.nan
#         zcols = rescols * 0.0
#         zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]
#===============================================================================


