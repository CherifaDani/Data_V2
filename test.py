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
#sys.path.append(r'/home/cluster')

script_path = 'deriv_script.xlsx'
var_name = 'FI_SWAP_EUR_1Y_LAST'
state_path = 'variable_state.csv'

b = Variable(script_path=script_path, state_path = state_path,
             var_name=var_name)

b.update()
