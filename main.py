# -*- coding: utf-8 -*-
# @PydevCodeAnalysisIgnore
import data_utils

from Variable import Variable
import derivation_functions

from datetime import datetime
import pandas as pd
import control_utils as cu
import csv
import data_utils
from os.path import basename
import os
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
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
graphviz = GraphvizOutput()
graphviz.output_file = 'basic.png'
# import derivation_calculs
sys.path.append(r'/home/cluster')
with PyCallGraph(output=graphviz):
script_path = 'CM.xlsx'
var_name = 'Corr_VIX_LAST-WTI'
var_name = 'Corr_GSCI-Gold'
var_name = 'STR_USD_1D'
var_name = 'Spread_CDS_CHN_5Y-CDS_USA_5Y'
var_name = 'STR_USD_1D_DACE_1_20_100'
state_path = 'x.csv'
state_path = 'variable_state.csv'
var_name = 'FUT_EURUSD_RET1C1'
var_name = 'STR_USD_1M'
var_name = 'Spread_SWAP_USD_1Y-GOV_USA_1Y'
var_name = 'FUT_BUND_RET1C1'
var_name = 'SP500_TREND6M'  # to test before very complex variable
var_name = 'FUT_BUND_TREND6M'  # complex variable ==> futures_roll
var_name = 'CDS_GER_1Y_Z250D'
var_name = 'CDS_GER_1Y'
var_name = 'STR_USD_1D_Z5D'



def processing_dir(dir_path1, dir_path2):
    # list_accepted = []
    list_dict = []
    element = ''
    for element in os.listdir(dir_path1):
        df1 = data_utils.load_var(join(dir_path1, element), '1')
        for element in os.listdir(dir_path2):
            df2 = data_utils.load_var(join(dir_path2, element), '2')
    print compare_two_dfs(df1, df2)


dir_path1 = 'I06'
dir_path2 = 'I04'
i = 0
j = 0
for element1 in os.listdir(dir_path1):
    for element2 in os.listdir(dir_path2):
        if element1 == element2:
            j += 1
            csv_path = dir_path1 + '/' + element1
            df_base = data_utils.load_var(csv_path, 'x')
            # df_base.columns = [x.lower() for x in df_base.columns]
            csv_path = dir_path2 + '/' + element2
            df_latest = data_utils.load_var(csv_path, 'y')
            dfs = [df_latest, df_base]

            if df_base.equals(df_latest):
                print(element1)
            else:
                i += 1
                print(element1, element2)
                print(compare_two_dfs(df_base, df_latest))
print(i)
print(j)
df_latest.sort_index(ascending=True, inplace=True)
df_latest.columns = [x.lower() for x in df_latest.columns]
df_final = pd.DataFrame()
df_final = df_base.append(df_latest)
df_final = df_final[~df_final.index.duplicated(take_last=False)]
print df_final.shape
df_final.to_csv(path + element1)

print processing_dir(dir_path1, dir_path2)


































