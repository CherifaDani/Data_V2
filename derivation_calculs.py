# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from data_utils import reindex
import numpy as np
import datetime


def apply_operation(var_list, freq, operation, parameters):
    """
    Function used to derive dataframes
 
    Parameters
    ----------
    var_list : {Objects list type}
                A list of variables (objects)
 
    freq : {Char type}
            The frequency of the Series,
            default: 'B'
 
    operation : {String type}
                The derivation to apply to the var_list
 
    parameters : {Dict type}
                 The parameters of the derivation

    Return
    ------
    fdf : {Dataframe type}
            full dataframes,
            Can be none if operation not found
    """
    fdf = pd.DataFrame()
    dfs = map(lambda x: read_df(x), var_list)
    if operation == 'timeshift':
        fdf = []
        for _, df in enumerate(dfs):
            df_calc = apply_timeshift(df, freq)
            fdf.append(df_calc)

        return fdf

    if operation == 'corr':
        # df_calc = apply_corr(df)
        fdf = dfs[0] + dfs[1]
        return fdf

    if operation == 'combi':
        # dfs = map(lambda x: read_df(x), var_list)
        coeff1 = parameters['coeff1']
        coeff2 = parameters['coeff2']
        islinear = parameters['lin']
        transfo = parameters['transfo'] if 'transfo' in parameters else None
        fdf = apply_combi(dfs[0], dfs[1], coeff1, coeff2, islinear, transfo)
        return fdf

    if operation == 'pctdelta':
        period = parameters['period']
        ownfreq = parameters['freq']
        fdf = apply_pctdelta(dfs[0], period=period, freq=ownfreq)
        return fdf

    if operation == 'rollingreturn':
        period = parameters['period']
        rollfreq = parameters['rollfreq']
        iweek = parameters['iweek']
        iday = parameters['iday']
        iroll_interval = parameters['iroll_interval']
        fdf = apply_rolling(dfs[0], dfs[1], rollfreq, iweek, iday, iroll_interval)

    if operation == 'ewma':
        emadecay  = parameters['emadecay']
        wres  = parameters['wres']
        wz  = parameters['wZ']
        fdf = apply_ewma(dfs[0], emadecay, wres, wz)






    if 'mult' in parameters:
        return fdf * parameters['mult']
    if 'lag' in parameters:
        return apply_timeshift(fdf, freq, parameters['lag'])
    if 'add' in parameters:
        return fdf + parameters['add']
        #===================================================================
            # f = lambda x, y: (x.read_var(x.get_param('path')) * )
            # dfs = map(f, var_list)
            # df_calc = map(lambda x, y: x*y, var_list)
            #===================================================================


def apply_timeshift(df, freq, shift=0):
        """
        Renvoie une copie de l'objet courant, avec dates translatées
        d'un délai.
        Les noms de colonnes de l'objet courant ne sont pas modifiés.
        freq représente l''unité de compte du décalage
        ownfreq représente la fréquence finale(propre) de la série.
        refdate: date de calcul. si fournie, les décalages sont limités
        à cette date
        Exemple: décaler de 20j une série trimestrielle
        """
        # Shiffting with a given shift
        ndf = df.tshift(shift, freq)
        return ndf


def apply_combi(df1, df2, coeff1, coeff2, islinear, transfo=None):
    dfs = pd.DataFrame()
    dfx, dfy = reindex(df1, df2)
    if islinear:
        dfs = coeff1 * dfx + coeff2 * dfy
    else:
        dfs = (coeff1 ** dfx) * (coeff2 ** dfy)

    if transfo is not None:
        if str(transfo).lower() == 'tanh':
            transfo_df = np.tanh(dfs)
        elif str(transfo).lower() == 'sign':
            transfo_df = np.sign(dfs)
        return transfo_df
    else:
        return dfs


def read_df(x):
    return x.read_var(x.get_param('path'))


def apply_corr(df,  period=1, span=20, exponential=True, inpct=True, lag=0):
    '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
           period: si 0, corrélation des valeurs, si > 0, corrélation des 
           variations sur period
           lag: retard sur la seconde colonne
           cols: spécifications de 1 ou 2 colonnes
        '''
    startval = period + lag * period
    cols = df.columns
    if len(cols) == 1:
        col1 = cols[0]
        col2 = col1
    else:
        col1 = cols[0]
        col2 = cols[1]

        if period == 0:
            data1 = df[col1]
            data2 = df[col2].shift(periods=lag)
        else:
            if inpct:
                data1 = df[col1].pct_change(period)[startval:]
                data2 = df[col2].pct_change(period).shift(
                    periods=lag * period)[startval:]
            else:
                data1 = df[col1].diff(period)[startval:]
                data2 = df[col2].diff(period).shift(
                    periods=lag * period)[startval:]

        if exponential:
            corrdata = pd.ewmcorr(data1[startval:],
                                  data2[startval:], span=span)
        else:
            corrdata = pd.rolling_corr(data1, data2, window=span)

        return corrdata


def apply_pctdelta(df, period, freq):
    deltadata = df.diff(period)
    idx_all = pd.bdate_range(start=(deltadata.index[0]).date(),
                             end=(deltadata.index[-1]).date(),
                             freq=freq)

    # Reindex using datetime index, to drop hours and minutes
    deltadata.index = pd.DatetimeIndex(deltadata.index).normalize()
    if(freq == 'B' or freq == 'D'):
        deltadata = deltadata.reindex(index=idx_all, method=None)

    else:
        deltadata = deltadata.reindex(index=idx_all, method='pad')

    return deltadata


def apply_rolling(df1, df2, rollfreq, iweek, iday, iroll_interval, freq):
    '''
        Renvoie la série des variations d'une colonne pour un décalage donné.
        Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
        Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
        où Vsubst représente la série self [substcol] 
        '''
    # élargir le calendrier pour inclure les dates de rolls de façon certaine
    idx_all = pd.bdate_range(start=(df1.index[0]).date(),
                             end=(df1.index[-1]).date(),
                             freq=freq)
    df1.index = pd.DatetimeIndex(df1.index).normalize()
    data = df1.reindex(index=idx_all, method=None)
    rolldates = pd.bdate_range(data.index[0], data.index[-1], freq=rollfreq)
    rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)


def apply_ewma(df, emadecay, wres, wz):
    '''Renvoie la série des ema d'un ensemble de colonnes pour une pseudo-durée(span) donnée.'''
    ''' self: contient la totalité des données primaires dont on veut calculer la moyenne
    emadecay: coefficient d'atténuation de la moyenne(proche de 1). Prioritaire si fourni.
    span: 2/emadecay - 1
    cols: groupe de colonnes dont on calcule l'ewma.
    wres: si True, on calcule également le résidu
    normalize: si True, on calcule aussi le Z-Score(résidu / ewmastd(même span))
    histoemadata: série facultative contenant les valeurs historiques de l'ewma sur des dates
       normalement antérieures aux données primaires.
    overridedepth: nombre de jours passés(à partir de la donnée la plus récente) à recalculer
    '''
    stdev_min = 1e-5
    rescols = pd.DataFrame()
    zcols = pd.DataFrame()
    # calculer la période synthétique correspondant au coeff s'il est fourni
    if type(emadecay) in [int, float]:
        if emadecay > 0:
            span = (2.0 / emadecay) - 1
    df_calc = pd.ewma(df, span=span, adjust=True)
    # calcul du résidu
    if wres:
        rescols = df - df_calc
        # calcul du ZScore
        if wz:
            stdevcols = pd.ewmstd(rescols, span=span)
            stdevcols[stdevcols <= stdev_min] = np.nan
            zcols = rescols * 0.0
            zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]

    return df, df_calc, rescols, zcols
#===============================================================================
# 
# 
# 
# 
# path = '/home/cluster/git/data_v2/2 Data/1 Received/Market data/Base/FDFD_Index_LAST_PRICE.csv'
# var_name = 'FI_STR_USD_1D_LAST'
# df= data_utils.load_var(path=path, var_name=var_name)
# # print df.head(5)
#===============================================================================

































