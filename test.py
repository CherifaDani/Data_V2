# -*- coding: utf-8 -*-
# @PydevCodeAnalysisIgnore
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
#sys.path.append(r'/home/cluster')

script_path = 'deriv_script.xlsx'
# var_name = 'Corr_VIX_LAST-WTI'
var_name = 'Corr_GSCI-Gold'
var_name2 = 'STR_USD_1D'
state_path = 'variable_state.csv'

b = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name)
c = Variable(script_path=script_path,
             state_path=state_path,
             var_name=var_name2)
print b.update()
#===============================================================================
# b.write_dict()
# c.write_dict()
# vlist = [b, c]
# parameters = {'mult': 200000 }
# freq = 'B'
# operation = 'timeshift'
# from derivation_calculs import ope_dict
# print derivation_calculs.apply_operation(vlist, freq, operation, parameters, ope_dict)
#===============================================================================
# print df1, df2
#===============================================================================
# map(lambda x: x.update(), vlist)
#===============================================================================
#print df1, df2
#print c.update() 
# print b.test()