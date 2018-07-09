import pandas as pd
import numpy as np
import derivation_calculs
import data_utils
import math
from pathlib2 import Path


def apply_vol(df, period, window, inpct, annualize, fillinit, freq):
    if period != 0:
        diffdata = derivation_calculs.apply_pctdelta(df, period, freq, inpct)
    else:
        diffdata = df
    voldata = pd.rolling_std(diffdata, window=window)
    if fillinit:
        voldata[0: window] = voldata[0: window + 1].fillna(method='bfill')
    voldata = voldata.dropna()
    cols = df.columns
    newcols = range(len(cols))
    for icol, col in enumerate(cols):
        if annualize:
            nfreqdict = derivation_calculs.estimate_nat_freq(col)
            nfreq = max(1, nfreqdict['min'])
            annfactor = math.sqrt(260 / nfreq)
        else:
            annfactor = 1
        newdf = voldata[voldata.columns[icol]] * annfactor

    return newdf

#
# df = data_utils.load_var('cds.csv', 'cds')
# params = {'period': 1, 'window': 20, 'inpct': True, 'annualize': False}
# dfx = apply_vol(df, period=params['period'], window=20, annualize=False, inpct=True, fillinit=True, freq='B')
# print(dfx)
path = 'nnn/ff/'

path = Path(path).with_suffix('.csv')
print(path)