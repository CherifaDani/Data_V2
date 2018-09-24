# -*- coding: utf-8 -*-

import data_utils
import matplotlib.pyplot as plt


# df1 = data_utils.load_var('2 Data/1 Received/Market data/Base/XAU_Curncy_LAST_PRICE.csv')  # commo_gold
# df1 = data_utils.load_var('2 Data/2 Calculs/18 06 Derived/I/STR_USD_1M.csv')  # str_usd_1m

# df1 = data_utils.load_var('usd_eur.csv')
df1 = data_utils.load_var('usd_eur_nan.csv')

plt.grid()
# plt.plot(df1, "r")
plt.plot(df1, 'r')
plt.show()
plt.figure()
