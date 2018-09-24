# -*- coding: utf-8 -*-

import data_utils


print data_utils.load_var('2 Data/2 Calculs/18 06 Derived/I/FUT_SP500_C1_RET1D.csv')  # rollingreturn

print data_utils.load_var('2 Data/2 Calculs/18 06 Derived/I/FUT_BUND_RET1ROLL.csv')  # futuresroll
print data_utils.load_var('2 Data/2 Calculs/18 06 Derived/I/FUT_NKY_RET1_STD50.csv')  # vol

path = '2 Data/2 Calculs/18 06 Derived/I/FUT_BUND_RET1ROLL.csv'  # futuresroll
print data_utils.load_var(path)
