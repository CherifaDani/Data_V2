# -*- coding: utf-8 -*-
# from __future__ import division
import pandas as pd
import numpy as np
import math
import logging

from datetime import datetime
from datetime import timedelta


try:
    import data_utils
except ImportError:
    data_utils = None
    raise ImportError("Don't find the package control_utils")

try:
    import var_logger
except ImportError:
    print ("Don't find the package var_logger")
    raise ImportError("Don't find the package var_logger")

# Launching the logger module
var_logger.setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
# Setting the DEBUG level of the logger
logger.setLevel(logging.INFO)


glbnano = 86400000000000.0
glbmetavar = '$$'
glbdefault_time = datetime(1900, 1, 1, 18, 30)


def extendtodate(df, todate=None, freq='B', limit=5):
    """
       Function used to extend a dataset to a newer date by extending the values

       Parameters
       ----------
       df : {Dataframe type}
            The input dataframe

       todate : {Datetime type}
                 A newer date
                 default: None

       freq : {Char type}
               The frequency of the dataframe,
               default: 'B'

       limit : {Integer type}
                The number of extended days not to exceed
                default: 5

       Return
       ------
       new_df : {Dataframe type}
                 The new dataframe
    """

    if todate is None:
        todate = datetime.now()
    else:
        todate = pd.to_datetime(todate)
    dt0 = df.index[-1] + timedelta(days=1)
    if dt0 <= todate:
        dtindex = pd.bdate_range(start=dt0, end=todate, freq=freq)
        newdf = pd.DataFrame(index=dtindex, columns=df.columns)
        for col in df.columns:
            newdf.ix[0, col] = df.ix[-1, col]
            newdf[col] = newdf[col].fillna(method='ffill', limit=limit)
        new_df = df.append(newdf)

    return new_df


def estimate_nat_freq(df, col):
    """
       Function used to estimate the natural frequency of a DF: the frequency of the change of the values

       Parameters
       ----------
       df : {Dataframe type}
            The input dataframe

       col : {Str or int type}

       Return
       ------
       new_df : {Dataframe type}
                 The new dataframe
    """
    df.dropna()
    df.sort_index(inplace=True)
    # fl = float((df.index.asi8[-1] - df.index.asi8[0]) / glbnano)
    fl = float(len(df.values))
    try:
        if type(col) == int:
            coldf = df[df.columns[col]]
        elif type(col) == str:
            coldf = df[col]
        else:
            return {}
    except Exception as e:
        return {}

    # série des différences
    ddf = coldf.diff(1)
    # série des différences non nulles
    ddf = df[ddf != 0]
    # rajouter une colonne pour les différences de dates
    ddf['deltat'] = 0
    # ddf.deltat[1:] = (ddf.index.asi8[1:] - ddf.index.asi8[: -1]) / glbnano
    # trier les intervalles entre changements de dates
    lastdelta = ddf.ix[-1]
    ddf.sort(columns='deltat', inplace=True)
    length = len(ddf)
    deltat = ddf.deltat[1:]
    fdict = {}

    if length > 1:
        fdict['last'] = lastdelta
        fdict['min'] = mind = deltat.min()
        fdict['datemin'] = deltat.idxmin()
        fdict['pct5'] = mind
        fdict['pct10'] = mind
        fdict['pct25'] = mind
        fdict['median'] = deltat.ix[int(0.5 * length) - 1]
        fdict['max'] = maxd = deltat.max()
        fdict['datemax'] = deltat.idxmax()
        fdict['pct95'] = maxd
        fdict['pct90'] = maxd
        fdict['pct75'] = maxd
        fdict['n1'] = len(deltat[deltat >= 1])
        fdict['r1'] = fdict['n1'] / fl
        fdict['n5'] = len(deltat[deltat >= 5])
        fdict['r5'] = fdict['n5'] / (fl / 5)
        fdict['n10'] = len(deltat[deltat >= 10])
        fdict['r10'] = fdict['n10'] / (fl / 10)
        fdict['n20'] = len(deltat[deltat >= 20])
        fdict['r20'] = fdict['n20'] / (fl / 20)
        if length > 4:
            fdict['pct25'] = deltat.ix[int(0.25 * length) - 1]
            fdict['pct75'] = deltat.ix[int(0.75 * length) - 1]
            if length > 10:
                fdict['pct10'] = deltat.ix[int(0.1 * length) - 1]
                fdict['pct90'] = deltat.ix[int(0.9 * length) - 1]
                if length > 20:
                    fdict['pct5'] = deltat.ix[int(0.05 * length) - 1]
                    fdict['pct95'] = deltat.ix[int(0.95 * length) - 1]
    return fdict


def get_columns(df, cols=None, forceuppercase=True):
    """
       Function used to return the column identifiers for an integer or a string

       Parameters
       ----------
       df : {Dataframe type}
            The input dataframe

       cols : {Str or int type}
              default: None

       forceuppercase : {Boolean type}
                         default: True
       Return
       ------
       new_df : {Dataframe type}
                 The new dataframe
    """
    if forceuppercase:
        cols_renamed = []
        for col in df.columns:
            cols_renamed += [col.upper()]

        df.columns = cols_renamed

    if cols is None:
        return df.columns
    else:
        retcolumns = []

        #  Cas où le type de la colonne est un entier
        if type(cols) == int:
            try:
                return [str(df.columns[cols])]
            except Exception as e:
                logstr = ' !! Interprétation du numéro de colonne de sortie : {} impossible !!'.format(str(cols))
                logger.exception(logstr)
                return []

        #  Cas où le type de la colonne n'est pas une liste mais une string/unicode
        elif type(cols) in [str, unicode]:
            try:
                if forceuppercase:
                    retcolumns = [str(cols).upper()]
                #  On la transforme en liste de string
                else:
                    retcolumns = [str(cols)]
                return retcolumns

            except Exception as e:
                logstr = ' !! Interprétation du nom de colonne de sortie : {} impossible !!'.format(str(cols))
                logger.exception(logstr)
                return []

        #  Si le type de la colonne d'entrée est déjà une liste, on
        elif type(cols) == list:

            ncols = len(df.columns)

            try:
                for col in cols:
                    if type(col) == int:
                        if col in range(0, ncols):
                            retcolumns.append(str(df.columns[col]))
                    elif type(col) in [str, unicode]:
                        if forceuppercase:
                            col = str(col).upper()
                        else:
                            col = str(col)

                        if col in df.columns:
                            retcolumns.append(col)
            except Exception as e:
                logstr = ' !! Interprétation de la liste de colonnes de sortie : {} impossible !!'.format(str(cols))
                logger.exception(logstr)
                return []

            return retcolumns

        #  Sinon on n'a ni des scalaires(entiers ou strings) ni une liste de colonnes
        else:
            logstr = ' !! Le type ' + str(type(cols)) + ' des colonnes de sortie :  {}  n\'est pas interprétable !!'.\
                        format(str(cols))
            logger.error(logstr)
            return []


def apply_timeshift(df, shift=1, freq='B', ownfreq=None, refdate=None):
    """
    This function returns a copy of the current dataframe, with translated dates of a delay(shift).

    Parameters
    ----------
    df : {Dataframe type}
          The input dataframe
    shift : {Integer type}

    freq : {Char type}
            The offset unit of the shift, default: 'B'
    ownfreq : {Char type}
               represents the final frequency of the DF
               default: None
    refdate : {Datetime type}
              Date of calculation. If provided,
              the offsets are limited to this date

    Return
    ------
    A shifted dataframe
    """
    new_df = df.copy()

    # pas de décalage: on peut changer la fréquence
    # if freq <> 'B':
    if ownfreq is not None and ownfreq != freq:
        pass
        # new_df=new_df.change_freq(freq=ownfreq)
    if shift == 0:
        return new_df

    if refdate is None:
        refdate = datetime.now()
    else:
        refdate = pd.to_datetime(refdate)

    ndf = new_df.tshift(shift, freq)

    # sous-typer en TDataSet
    # Vérifier que l'on ne décale pas au-delà d'aujourd'hui
    lastdate = ndf.index[-1]
    if lastdate > refdate:
        # lastline=ndf.ix [-1]
        newline = ndf.ix[[-1]]
        ndf = ndf[ndf.index < refdate]
        ndf = ndf.append(newline)

    ndf.columns = ['SHIFT']
    return ndf


def apply_combi(df1, df2, coeff1=1, coeff2=0, constant=0,
                islinear=True, transfo=None):
    """
    This function returns  the linear or exponential combination of two columns

    Parameters
    ----------
    df1 : {Dataframe type}
          The input dataframe

    df2 : {Dataframe type}
          The input dataframe

    coeff1 : {Float type}
              A multiplicative coefficient if islinear is True,
              else exponential, applied to the first DF
              Default: 1

    coeff2 : {Float type}
              A multiplicative coefficient if islinear is True,
              else exponential, applied to the second DF
              Default: 0

    constant : {Float type}
               This value is added if islinear is true, else multiplied.
                Default: 0

    islinear : {Boolean type}
               Designates the nature of the operation
               True: addition
               False: multiplication
               default: None

    Return
    ------
    new_df : The output dataframe
    """
    df = pd.concat([df1, df2], axis=1)
    df = df.fillna(method='ffill')
    # df = df1.merge(df2, left_index=True, right_index=True)

    df.columns = ['df1', 'df2']
    cols1 = get_columns(df)

    if len(cols1) > 0:
        col1 = cols1[0]
        print(col1)
        datacol1 = df[col1]

    else:
        datacol1 = None

    cols2 = get_columns(df)
    if len(cols2) > 0:
        col2 = cols2[1]
        datacol2 = df[col2]

    else:
        datacol2 = None
    c1null = c1one = c1neg = c2null = c2one = c2neg = False
    if coeff1 == 0 or datacol1 is None:
        c1null = True
    elif abs(coeff1) == 1:
        c1one = True
    if coeff1 < 0:
        c1neg = True
    if coeff2 == 0 or datacol2 is None:
        c2null = True
    elif abs(coeff2) == 1:
        c2one = True
    if coeff2 < 0:
        c2neg = True

    if islinear:
        combiarray = np.zeros(len(df.index)) + constant
        if not c1null:
            combiarray = datacol1.values * coeff1 + constant

        if not c2null:
            combiarray = combiarray + datacol2.values * coeff2

    else:

        # constante égale à 0 en multiplicatif: on la prend pour 1
        if constant == 0:
            constant = 1
        combiarray = np.ones(len(df.index)) * constant

        if (datacol1 is not None):
            combiarray = np.power(datacol1.values, coeff1) * constant
        if (datacol2 is not None):
            combiarray = combiarray * np.power(datacol2.values, coeff2)
    if transfo is not None:
        if str(transfo).lower() == 'tanh':
            combiarray = np.tanh(combiarray)
        elif str(transfo).lower() == 'sign':
            combiarray = np.sign(combiarray)
    new_df = pd.DataFrame(index=df.index, data=combiarray)
    new_df.columns = ['COMBI']
    return new_df


def take_columns(df, cols=None, forceuppercase=True):
    """
    This function returns one or many columns of a DF

    Parameters
    ----------
    df : {Dataframe type}
          The input dataframe

    cols : {List type}
            A list of the columns
            Default: None

    forceuppercase : {Boolean type}
                      Default: True

    Return
    ------
    ds : The output dataframe
    """
    if cols is None:
        return df
    columns = get_columns(df, cols=cols, forceuppercase=forceuppercase)
    if len(columns) > 0:
        try:
            ds = pd.DataFrame(index=df.index,
                              data=df[columns],
                              columns=columns)

        except Exception as e:
            return None
    else:
        ds = None
    return ds


def apply_ewma(df, emadecay=None, span=1, inplace=True,
               cols=None, wres=True, normalize=True,
               histoemadata=None, overridedepth=0,
               stdev_min=1e-5):
    """
    This function returns the ema series of a set of columns for a given pseudo-span

    Parameters
    ----------
    df : {Dataframe type}
          The input dataframe

    emadecay : {Float type}
                Attenuation coefficient of the mean (close to 1)
                Default: None

    span : {Float type}
            span = 2/emadecay - 1
            Default: 1

    inplace : {Boolean type}
            Default: True

    cols : {List type}
            A list of columns whose ewma is calculated
            Default: None

    wres : {boolean type}
            If True, the residue is also calculated
            Default: True

    normalize : {Boolean type}
                If True, The Z-score: (residu / ewmastd) is calculated
                 Default: True

    histoemadata : {Integer type}
                    Optional series containing the historical values ​​of the ewma on dates
                    normally prior to the primary data.
                     Default: None

    overridedepth : {Integer type}
                    The number of days spent (from the most recent data) to recalculate
                    Default: 0

    stdev_min : {Float type}
                A minimum tolerated value for the exponentially-weighted moving std
                Default: 1e-5

    Return
    ------
    new_df : The output dataframe
    """
    usehistoforinit = False
    if (histoemadata is not None) \
            and (type(histoemadata) == type(df)) \
            and (len(histoemadata.index) > 0) \
            and np.array_equiv(df.columns, histoemadata.columns):
        if (histoemadata.index[0] <= df.index[-1 - overridedepth]) and (
                histoemadata.index[-1] >= df.index[-1 - overridedepth]):
            usehistoforinit = True

    df.sort_index(inplace=True)
    if usehistoforinit:
        # cas où on fournit un historique des ema
        histoemadata.sort_index(inplace=True)

    cols = get_columns(df, cols)
    # cols = df.columns
    # if not(col in self.columns) : return new_df
    # import pdb; pdb.set_trace()
    # extraction des données à moyenner dans un DataFrame
    datacols = pd.DataFrame(data=df[cols])
    if inplace:
        new_df = df
    else:
        new_df = df.copy()
        new_df = new_df.take_columns(cols)
    # calculer la période synthétique correspondant au coeff s'il est fourni
    if type(emadecay) in [int, float]:
        if emadecay > 0:
            span = 2.0 / emadecay - 1

    if usehistoforinit:
        # historique d'ema fourni
        dhistolast = histoemadata.index[-1]
        dnewlast = df.index[-1]
        # si plus d'historique que de données nouvelles, rien à faire
        if dhistolast >= dnewlast:
            return histoemadata
        if type(dhistolast) == int:
            dfirstnewema = dhistolast + 1
        else:
            dfirstnewema = dhistolast + timedelta(days=1)
        # extraction du segment de nouvelles données
        datacols = datacols.ix[dfirstnewema: dnewlast]
        # calcul de l'ema
        newemadata = pd.ewma(datacols, span=span, wres=wres, normalize=normalize)
        # recollement des nouvelles données
        emadata = histoemadata
        emadata.patch_append(newemadata, check_overlap=True)
    else:
        # recalcul de la totalité des données de l'ema
        emadata = pd.ewma(datacols, span=span, adjust=True)
        new_df = emadata
    # calcul du résidu
    if wres:
        rescols = df[cols] - emadata
        # calcul du ZScore
        if normalize:
            stdevcols = pd.ewmstd(rescols, span=span)
            stdevcols[stdevcols <= stdev_min] = np.nan
            zcols = rescols * 0.0
            zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]
        for col in cols:
            new_df[col] = emadata[col]
            if wres:
                new_df[col] = rescols[col]
                if normalize:
                    new_df[col] = zcols[col]
    return new_df


def apply_corr(df1, df2, period=1,
               span=20, exponential=True,
               inpct=True, lag=0):
    """
        This function computes the correlation between two DFs

        Parameters
        ----------
        df1 : {Dataframe type}
              The input dataframe

        df2 : {Dataframe type}
              The input dataframe


        period : {Integer type}
                 If period = 0: Apply correlation between two DFs
                 If period > 0: Apply correlation of variations over period
                 Default: 1

        exponential : {Boolean type}
                    Default: True

        span : {Integer type}
                The rolling window size
                Default: 20

        inpct : {Boolean type}
                Use of arithmetic or geometric returns
                Default: True

        lag : {Integer type}
              Delay on the second column
              Default: 0

        Return
        ------
        new_df : The output dataframe

    """
    # hm_time = df2.index[0]
    #
    # df1 = set_daytime(df1, hm_time=hm_time)
    # # print(df1)
    # df1.columns = ['df']
    # df2.columns = ['df']
    idx_all = pd.bdate_range(start=df2.index[0], end=df2.index[-1], freq='B')
    # effacer l'heure pour synchronisation
    df1 = set_daytime(df1, datetime(2000, 1, 1))
    df2 = set_daytime(df2, datetime(2000, 1, 1))
    # élargir le calendrier pour inclure les dates de rolls de façon certaine
    # df1 = df1.reindex(index=idx_all, method=None)
    df = pd.concat([df1, df2], axis=1)
    df = df.fillna(method='ffill')
    df.columns = ['df1', 'df2']

    cols = get_columns(df)
    if len(cols) == 1:
        col1 = cols[0]
        col2 = col1

    else:
        col1 = cols[0]
        col2 = cols[1]

    # #  si period == 0 c'est l'autocorrélation des valeurs
    # #  et non des variations qui est calculée
    startval = period + lag * period
    if period == 0:
        data1 = df[col1]
        data2 = df[col2].shift(periods=lag)

    else:
        if inpct:
            data1 = df[col1].pct_change(period)[startval:]

            data2 = df[col2].pct_change(period).shift(periods=lag * period)[startval:]

        else:
            data1 = df[col1].diff(period)[startval:]
            data2 = df[col2].diff(period).shift(periods=lag * period)[startval:]

    if exponential:
        corrdata = pd.ewmcorr(data1[startval:], data2[startval:], span=span)
    else:
        corrdata = pd.rolling_corr(data1, data2, window=span)

    new_df = pd.DataFrame(index=df.index, data=(corrdata))
    new_df.columns = ['CORR']

    return new_df

# def apply_corr(dfx, dfy, span=20,  period=1,
#                exponential=True,
#                inpct=True, cols=None, lag=0
#                ):
#     '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
#        period: si 0, corrélation des valeurs, si > 0, corrélation des variations sur period
#        lag2: retard sur la seconde colonne
#        cols: spécifications de 1 ou 2 colonnes
#     '''
#     if dfy is not None:
#         [df1, df2] = reindex(dfx, dfy)
#     else:
#         df1 = dfx
#         df2 = dfx
#     # print(df1.shape)
#     new_df = pd.DataFrame(index=dfx.index)
#     col1 = df1
#     col2 = df2
#     # if len(cols) == 1:
#     #     col1 = df1
#     #     col2 = col1
#     # else:
#     #     col1 = df1
#     #     col2 = df2
#     # #  si period == 0 c'est l'autocorrélation des valeurs
#     # #  et non des variations qui est calculée
#     startval = period + lag * period
#     if period == 0:
#         data1 = col1
#         data2 = col2.shift(periods=lag)
#     else:
#         if inpct:
#             data1 = col1.pct_change(period)[startval:]
#             data2 = col2.pct_change(period).shift(periods=lag * period)[startval:]
#         else:
#             data1 = col1.diff(period)[startval:]
#             data2 = col2.diff(period).shift(periods=lag * period)[startval:]
#     if exponential:
#         corrdata = pd.ewmcorr(data1[startval:], data2[startval:], span=span)
#     else:
#         corrdata = pd.rolling_corr(data1, data2, window=span)
#
#     corrdata = corrdata.dropna()
#     # new_df['CORR'] = corrdata
#     new_df = corrdata
#     new_df.columns = ['CORR']
#
#     # print(new_df)
#     return new_df


def apply_vol(df,
              period=1,
              window=20, inplace=True,
              annualize=True,
              fillinit=True,
              inpct=True,
              cols=None
              ):
    """
         This function returns the return volatility series

         Parameters
         ----------
         df : {Dataframe type}
               The input dataframe

         period : {Integer type}
                  If period = 0: Apply correlation between two DFs
                  If period > 0: Apply correlation of variations over period
                  Default: 1

         inplace : {Boolean type}
                     Default: True

         window : {Integer type}
                 The rolling window size
                 Default: 20

         inpct : {Boolean type}
                 Use of arithmetic or geometric returns
                 Default: True

         annualize : {Boolean type}
                      Annualized volatility if True, else dependant on the size of the window
                      Default: True

         fillinit : {Boolean type}
                    If True, copy of the first dates inside the first window of calculation
                    Default: True

         cols : {List type}
                Default: None

         Return
         ------
         new_df : The output dataframe

     """
    if inplace:
        new_df = df
    else:
        new_df = df.copy()
    if cols is None:
        cols = df.columns.values
        cols = data_utils.check_cell(cols)
    cols = get_columns(df, cols)
    # attention, le diff peut changer l'index
    if period == 0:
        diffdata = take_columns(df, cols)
    else:
        diffdata = take_diff(df, period=period, cols=cols, inpct=inpct, alldays=False)

    voldata = pd.rolling_std(diffdata, window=window)
    if fillinit:
        voldata[0: window] = voldata[0: window + 1].fillna(method='bfill')
    voldata = voldata.dropna()
    # pdb.set_trace()
    # voldata=pd.rolling_mean(diffdata, window=window)
    # newcols = range(len(cols))
    for icol, col in enumerate(cols):
        if annualize:
            nfreqdict = estimate_nat_freq(df, col)
            nfreq = max(1, nfreqdict['min'])
            annfactor = math.sqrt(260 / nfreq)
        else:
            annfactor = 1
        new_df[col] = voldata[voldata.columns[icol]] * annfactor

    # new_df.columns = newcols
    # new_df.columns = ['VOL']
    print(new_df)
    return new_df


def take_diff(df, period=1, inplace=False, cols=None, inpct=True,
              alldays=True, ownfreq=None):
    """
         This function returns the series of differences of a column for a given offset

         Parameters
         ----------
         df : {Dataframe type}
               The input dataframe

         period : {Integer type}

                  Default: 1

         inplace : {Boolean type}
                    If True, returns a copy of the DF
                   Default: False

         alldays : {Boolean type}
                    Default: True

         ownfreq : {Boolean type}
                    The natural frequency of the series
                    Default: None

         cols : {List type}
                Default: None

         Return
         ------
         new_df : The output dataframe

     """

    if inplace:
        new_df = df
    else:
        new_df = None

    cols = get_columns(df, cols)
    # if not(col in self.columns) : return new_df
    # import pdb; pdb.set_trace()
    # datacols=pd.DataFrame(data=self [cols], columns=cols)
    datacols = df[cols]

    # l'instruction suivante renvoie un DataFrame
    # Calculer la série des différences dans l'unité naturelle de la série
    if not inpct:
        # deltadata=datacols.diff(period)
        deltadata = datacols.diff(period)
    else:
        # deltadata=datacols.pct_change(period)
        deltadata = datacols.pct_change(period)

    # dsdelta=TDataSet(index=deltadata.index, data=deltadata)
    if alldays and ownfreq is not None:
        # zerotime=datetime.time(hour=0, minute=0, second=0)
        # dsdelta.set_daytime(zerotime)
        #  Prendre les dates sans heure de la série d'origine
        deltadata.index = deltadata.index.map(
            lambda (x): datetime(year=x.year, month=x.month, day=x.day,
                                 hour=0, minute=0))
        idx_all = pd.bdate_range(start=df.index[0], end=df.index[-1], freq=ownfreq)
        if (ownfreq == 'B' or ownfreq == 'D'):
            # pas de remplissage
            # deltadata=deltadata.reindex(index=idx_all, fill_value=0.0)
            deltadata = deltadata.reindex(index=idx_all, method=None)
        elif (ownfreq != 'B' and ownfreq != 'D'):
            # cas d'une série mensuelle suréchantillonnée quotidiennement:
            #  on prolonge la dernière variation calculée

            deltadata = deltadata.reindex(index=idx_all, method='pad')
        else:
            pass

    # import pdb; pdb.set_trace()
    #  en cas de copie d'objet: on ne renvoie que la colonne résultat
    # newcols = range(len(cols))

    if inplace:
        for col in cols:
            new_df[col] = deltadata[col]
    else:
        # si l'objet diffseries est un pd.Series, il n'a pas de champ columns
        new_df = pd.DataFrame(index=deltadata.index,
                                  data=deltadata.values,
                                  columns=cols)
    # new_df.columns = newcols
    # new_df = new_df.asfreq(freq='B')
    # new_df.reindex(index=idx_all)
    new_df.columns = ['X']
    return new_df


def apply_rolling(maincol, substcol, iday=1, iweek=1, rollfreq='BMS', effectiveroll_lag=0, inpct=True):

    """
     This function returns the series of variations of a column for a given offset.

     Parameters
     ----------
     maincol : {Dataframe type}
                The future c1

     substcol : {Dataframe type}
                 The future c2

     rollfreq : {String type}
                Default: 'BMS' => Business Month Start

     iday : {Integer type}
            The day number where the roll of the contract
            Default: 1

     iweek : {Integer type}
            The week number where the next future (c1) expires
             Default: 1

     effectiveroll_lag : {Integer type}
                          Can be 0 or 1;
                          It indicates whether the future c1 is used until the included roll date (value at 0)
                          or until the previous day (value at 1)
                          Default: 0

     inpct : {Boolean type}
              Creation of future prices from arithmetic (inpct=False)
              or geometric (inpct=True) returns
              Default: True

     Return
     ------
     new_df : The output dataframe

     """

    assert type(rollfreq) == str
    assert iday >= 0
    assert iweek >= 0
    period = 1
    assert effectiveroll_lag in [0, 1]
    df = pd.concat([maincol, substcol], axis=1)
    df = df.fillna(method='ffill')
    cols = get_columns(df)
    if cols is None:
        return None

    maincol = cols[0]
    substcol = cols[1]
    datacols = df[maincol]

    if not inpct:
        retdata = pd.DataFrame(datacols.diff(period))
    else:
        retdata = pd.DataFrame(datacols.pct_change(period))
    idx_all = pd.bdate_range(start=df.index[0], end=df.index[-1], freq='B')
    # effacer l'heure pour synchronisation
    retdata = set_daytime(retdata, datetime(2000, 1, 1))
    df = set_daytime(df, datetime(2000, 1, 1))
    # élargir le calendrier pour inclure les dates de rolls de façon certaine
    retdata = retdata.reindex(index=idx_all, method=None)

    # générer la série des dates de roll
    #          if rollfreq [1:].find('BS') < 0:
    #              rollfreq=rollfreq + 'BS'

    #     Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
    #     Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
    #     où Vsubst représente la série substcol

    rolldates = pd.bdate_range(start=df.index[0], end=df.index[-1], freq=rollfreq)
    rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)
    # Ne garder que les dates de roll antérieures aux données courantes
    rolldates = rolldates[rolldates <= retdata.index[-1]]
    daybefore_rolldates = rolldates + pd.datetools.BDay(-period)
    dayafter_rolldates = rolldates + pd.datetools.BDay(period)

    # timeidx=self.index
    # Contrat M(front) coté jusqu'à TRoll, traité jusqu'en TRoll-1, roulé en TRoll-1
    # Contrat M+1(next), coté jusqu'à TRoll, devient le front en TRoll + 1, traité en TRoll-1
    # Returns:
    #  en TRoll, Close(F2, TRoll)/ Close(F2, TRoll-1) - 1
    #  en TRoll + 1, Close(F1, TRoll+1)/ Close(F2, TRoll-1) - 1
    # dayafter_roll_contract=maincol
    # cas de UX
    if effectiveroll_lag == 0:
        roll_contract = maincol
        daybefore_roll_contract = maincol
    # cas de FVS
    else:
        roll_contract = substcol
        daybefore_roll_contract = substcol

    if inpct:
        rollday_returns = df.loc[rolldates, roll_contract].values / \
                          df.loc[daybefore_rolldates, daybefore_roll_contract].values - 1
        dayafter_returns = df.loc[dayafter_rolldates, maincol].values / \
                           df.loc[rolldates, substcol].values - 1
    else:
        rollday_returns = df.loc[rolldates, roll_contract].values - \
                          df.loc[daybefore_rolldates, daybefore_roll_contract].values
        dayafter_returns = df.loc[dayafter_rolldates, maincol].values - \
                           df.loc[rolldates, substcol].values

    newcol = 'ROLLING_RETURN'
    retdata.loc[rolldates, maincol] = rollday_returns
    retdata.loc[dayafter_rolldates, maincol] = dayafter_returns
    new_df = pd.DataFrame(index=retdata.index,
                              data=retdata.values,
                              columns=[newcol])
    # revenir au calendrier restreint
    new_df = pd.DataFrame(new_df.dropna())
    new_df.columns = [newcol]
    # new_df.name = retdataname
    return new_df


def set_daytime(df, hm_time, dates=None):
    """
     This function sets the time of day for the entire index or for specific dates.

     Parameters
     ----------
     df : {Dataframe type}
           The input dataframe

     hm_time : {Datetime or Timestamp type}
                Hour / minute / second to apply to dates

     dates : {List type}
              List of dates to modify

     Return
     ------
     new_df : The output dataframe, with a new index

     """
    if type(hm_time) in [datetime, pd.tslib.Timestamp]:
        if dates is None:
            newidx = df.index.map(lambda (x): datetime(year=x.year,
                                                     month=x.month,
                                                     day=x.day,
                                                     hour=hm_time.hour,
                                                     minute=hm_time.minute,
                                                     second=hm_time.second))

        else:
            if type(dates) != list:
                dates = [dates]
                newidx = df.loc[dates].index.map(lambda (x): datetime(year=x.year,
                                                                               month=x.month,
                                                                               day=x.day,
                                                                               hour=hm_time.hour,
                                                                               minute=hm_time.minute,
                                                                               second=hm_time.second))
        df.index = newidx
        return df


def apply_lag(df, lag=1, freq='B', cols=None, inplace=False):

    """
     This function returns a copy of the current object, with values ​​lagged of a delay and realigned

     Parameters
     ----------
     df : {Dataframe type}
           The input dataframe

     lag : {Integer type}
            offset time (delay)
            Default: 1

     freq : {Char type}
            Default: 'B'

     cols  : {List type}
             Default: None

     inplace : {Boolean type}
                Default: False

     Return
     ------
     new_df : The output dataframe, with a new index

     """
    new_df = df.copy()
    if lag == 0:
        return new_df

    cols = get_columns(df, cols)
    # if not(cols in self.columns) : return new_df
    dfcols = pd.DataFrame(data=df[cols], columns=cols)
    # la série décalée
    # import pdb; pdb.set_trace()
    laggedseries = dfcols.shift(periods=lag, freq=freq, inplace=inplace)

    # drop duplicates dates
    index_name = laggedseries.index.name
    if index_name is None:
        index_name = 'index'
        laggedseries.index.name = 'index'
    laggedseries = laggedseries.reset_index().drop_duplicates(subset=index_name, take_last=True).set_index(
        index_name)
    #  en cas de copie: on renvoie un dataset contenant les colonne décalées
    if inplace:
        for col in cols:
            new_df[col] = laggedseries[col]
    else:
        new_df = pd.DataFrame(index=laggedseries.index,
                              data=laggedseries.values,
                              columns=cols)

    # idx_all = pd.bdate_range(start=df.index[0], end=df.index[-1], freq=freq)
    # new_df = new_df.reindex(index=idx_all, method=None)
    # bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # new_df = new_df.resample(bday_us)
    # # print('newda')
    # print(new_df)
    # new_df.to_csv('vv.csv')
    return new_df


def apply_cumulative_return(df, timeweight=False, cols=None, inplace=True):

    """
     This function returns a compound cumulative returns

     Parameters
     ----------
     df : {Dataframe type}
           The input dataframe

     timeweight : {Boolean type}
                  Takes into account the duration between each date
                  expressed in the natural frequency of the DF
                   Default: False

     cols  : {List type}
             Default: None

     inplace : {Boolean type}
                Default: False

     Return
     ------
     new_df : The output dataframe, with a new index

     """
    cols = get_columns(df, cols)
    # pdb.set_trace()
    # datacols=pd.DataFrame(data=self [cols])
    if timeweight is True:
        deltatime = pd.Series(df.index.asi8)
        deltatime = deltatime.diff(1) / glbnano
        deltatime.fillna(value=0.0, inplace=True)
        deltatime = deltatime / 365.25
        deltatime = deltatime.reshape(len(df.index), 1)
        df[cols] = df[cols] * deltatime
    navdata = np.log(1 + df[cols])
    navdata = pd.expanding_sum(navdata)
    navdata = np.exp(navdata)
    newcols = range(len(cols))

    if inplace:
        new_df = df
    else:
        new_df = pd.DataFrame(data=navdata)
    for icol, col in enumerate(cols):
        if inplace:
            new_df[col] = navdata[col]

    new_df.columns = newcols

    return new_df


def apply_futures_roll(col_c1, col_c2, roll_dict):

    """
     This function computes the roll_adjusted returns on a series of first and second futures,
     for a rolling rule

     Parameters
     ----------
     col_c1 : {Dataframe type}
              The dataframe storing contract 1 prices

     col_c2 : {Dataframe type}
              The dataframe storing contract 2 prices

     roll_dict : {Dict type}
                 A dictionary describing the rolling rule

     Return
     ------
     new_df : The output dataframe, with the following columns:
                C1: the price of c1 contract
                C1_ROLL: the synthetic price of roll-adjusted c1 contract
                RETURN_1D_AFTER_ROLL: the daily roll-adjusted return

    """
    df = pd.concat([col_c1, col_c2], axis=1)
    df = df.fillna(method='ffill')
    # df.dropna(inplace=True)
    h_c1, m_c1, s_c1 = df.index[0].hour, df.index[0].minute, df.index[0].second
    dstart = df.index[0].replace(hour=h_c1, minute=m_c1, second=s_c1)
    dend = df.index[-1].replace(hour=h_c1, minute=m_c1, second=s_c1)
    bdates = pd.bdate_range(dstart, dend, freq='B')
    bdates = map(lambda d: d.replace(hour=h_c1, minute=m_c1, second=s_c1), bdates)
    # les dates de roll
    df_rolldates = apply_roll_shift(bdates, roll_dict)
    # # # # #  les rendements quotidiens
    df_ret1 = take_columns(col_c1).dropna().pct_change(periods=1)
    df_ret1.columns = ['RETURN_1D']
    # rendements modifiés après prise en compte du roll
    df_ret1['RETURN_1D_AFTER_ROLL'] = df_ret1['RETURN_1D']
    # le jour suivant le roll dans le calendrier du contrat
    df_ret1['NEXT_BDATE'] = np.nan
    df_ret1['NEXT_BDATE'][:-1] = df_ret1.index[1:]
    # les cours des 2 premiers contrats
    df_ret1['C1'] = take_columns(col_c1).dropna()
    df_ret1['C2'] = take_columns(col_c2).dropna()
    # le ratio c1 / c2 prolongé à tous les jours cotés
    df_ret1['RATIO12'] = (df_ret1['C1'] / df_ret1['C2']).fillna(method='ffill')
    next_bd_after_roll = df_ret1.loc[df_rolldates['LAST_TRADING_DATE'],
                                     'NEXT_BDATE'].fillna(method='ffill').fillna(method='bfill')
    df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = 1.0 + df_ret1.loc[
        next_bd_after_roll, 'RETURN_1D_AFTER_ROLL']
    df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] *= df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] * df_ret1.loc[df_rolldates['LAST_TRADING_DATE'], 'RATIO12'].values
    df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = df_ret1.loc[
                                                                  next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] - 1.0

    df_roll = pd.DataFrame(index=df_ret1.index,
                           columns=['C1', 'C1_ROLL', 'RETURN_1D_AFTER_ROLL', 'RETURN_1D', 'C2'])
    df_roll['C1_ROLL'] = 1.0

    roll_ret = np.log(1 + df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL'])
    roll_ret = pd.expanding_sum(roll_ret)
    roll_nav = np.exp(roll_ret)
    df_roll['C1_ROLL'] = roll_nav * df.iloc[0, 0]
    df_roll['C1'] = take_columns(col_c1)
    df_roll['RETURN_1D_AFTER_ROLL'] = df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL']
    df_roll['RETURN_1D'] = df_ret1.loc[:, 'RETURN_1D']
    df_roll['C2'] = df_ret1.loc[:, 'C2']

    return pd.DataFrame(df_roll)


def apply_roll_shift(dates, roll_dict):

    """
         This function returns a series of rolling dates from an initial list of reference dates and a dictionary

         Parameters
         ----------
         dates : {List type}
                  A list of dates

         roll_dict : {Dictionary type}
                      A dictionary with the following entries
                     'freq': the frequency of rolling dates, default 'BQ'
                     'day': if specified, the hard-coded calendar day in month of the rolling date,
                            default -1(not specified)
                     'week': if day==-1, the index of week roll in month(starting at 0),
                             default 0
                     'weekday': if day==-1, the index of day in week(starting at 0 for Mondays)
                     'bday_offset': an extra shift in business days, generally negative or zero,
                                    default 0
                     'bmonth_offset': an extra shift in months, generally negative or zero,
                                    default 0
         
         Return
         ------
         rolldates : The list of last rolling dates preceding each element of the given reference dates

         """
    if type(dates) is not list:
        dates = [dates]

    roll_freq = roll_dict['freq']
    # les dates de fin de périodes(mois ou trimestre ouvrï¿½)
    b_enddates = pd.bdate_range(dates[0], dates[-1], freq=roll_freq)
    b_enddates = map(lambda d: d.replace(hour=dates[0].hour,
                                         minute=dates[0].minute,
                                         second=dates[0].second), b_enddates)
    rollrule = pd.datetools.WeekOfMonth(weekday=roll_dict['weekday'],
                                        week=roll_dict['week'])
    bdayrollrule = pd.datetools.BDay(n=roll_dict['bday_offset'])
    monthrollrule = pd.datetools.BMonthEnd(n=roll_dict['bmonth_offset'])
    roll_day = roll_dict['day']

    # les dates de roll
    rolldates = pd.DataFrame(index=b_enddates, columns=['LAST_TRADING_DATE'])
    rolldates['LAST_TRADING_DATE'] = b_enddates
    if roll_dict['bmonth_offset'] != 0:
        rolldates['LAST_TRADING_DATE'] = map(lambda d: monthrollrule.rollback(d),
                                                rolldates['LAST_TRADING_DATE'])
    if roll_day >= 0:
        rolldates['LAST_TRADING_DATE'] = map(lambda d: d.replace(day=roll_day),
                                                rolldates['LAST_TRADING_DATE'])
    else:
        rolldates['LAST_TRADING_DATE'] = map(lambda d: rollrule.rollback(d),
                                                rolldates['LAST_TRADING_DATE'])

    if roll_dict['bday_offset'] != 0:
        rolldates['LAST_TRADING_DATE'] = map(lambda d: d + bdayrollrule,
                                                rolldates['LAST_TRADING_DATE'])
    rolldates.dropna(inplace=True)
    return rolldates


def fill_missing_values(idxmain, idxsubst, dfsubst=None):
    """
    This function fills the missing values ​​of the idxmain column with the idxsubst column
    Parameters
    ----------
    col_c1 : {Dataframe type}
             The dataframe storing contract 1 prices

    col_c2 : {Dataframe type}
             The dataframe storing contract 2 prices

    roll_dict : {Dict type}
                A dictionary describing the rolling rule

    Return
    ------
    new_df : The output dataframe

       """
    df = pd.concat([idxmain, idxsubst], axis=1)
    if dfsubst is None:
        df2 = df
    else:
        df2 = dfsubst
    try:
        maincol = get_columns(idxmain)[0]
        substcol = get_columns(df2)[0]
        if dfsubst is not None:
            df[substcol] = df2[substcol]

        df[maincol][pd.isnull(df[maincol])] = \
            df[substcol][pd.isnull(df[maincol])]
    except Exception as e:
        pass
    return df


def apply_ohlc_vol(df, OHLCcols=None,
                       window=20, inpct=True,
                     annualize=True,
                     fillinit=True,
                     algo='yang'):
    '''Renvoie la série des volatilités de rendements '''
    pass


def auto_categorize(df, mod=10, level_date=None, date_end=None, min_r=0.02):
    """
    This function returns a list: categorized DF
    or None if we have less than two modalities,
    bins are the number of modalities

    Parameters
    ----------
    df : {Dataframe type}
             The input dataframe

    mod : {Dataframe type}
             The dataframe storing contract 2 prices
            Default: 10

    level_date : {Datetime type}
                A dictionary describing the rolling rule
                Default: None

    date_end : {Datetime type}
                Default: None

    min_r : {Float type}
            Default: 0.02

    Return
    ------
    new_df : The output dataframe

       """
    df_copy = df.copy()
    if date_end is not None:
        if ds_copy.index.nlevels == 1:
            ds_copy = ds_copy.loc[:date_end]
        # cas d'un multi index
        elif ds_copy.index.nlevels == 2:
            ds_copy = ds_copy.loc[ds_copy.index.get_level_values(level_date) <= date_end]
            ds_copy = ds_copy.stack()

    df_q = [ds_copy.quantile(q=i / float(mod)) for i in range(0, int(mod) + 1, 1)]
    df_q = pd.DataFrame(df_q)
    bins = list(np.unique(df_q.dropna(how='all')))

    if len(bins) > 2:
        bins[0] = -10000000.
        bins[-1] = 10000000.

        def qq(serie, bins):
            '''catégorise les colonnes d'un DF '''
            res = pd.cut(serie, bins=bins, right=False, retbins=True, labels=False)[0]
            return res

        def check_df(df, min_r=min_r, bins=bins):
            # vérification si toutes le modalitées couvre au moins minR%.
            # Si ok on renvoi les bins d'origine, sinon on finsionne
            # les modalités et en renvoi les bins
            bins_ = bins[:]
            total = float(len(df.dropna(how='all')))
            for i in range(len(bins) - 1):
                j = i + 1
                try:
                    v = len(df.loc[(df >= bins[i]) & (df < bins[j])].dropna(how='all'))
                except Exception as e:
                    col_ = df.columns[0]
                    v = len(df[col_].loc[(df[col_] >= bins[i]) & (df[col_] < bins[j])].dropna())
                if (v / total) < min_r:
                    bins_.remove(bins[j])
            return bins_

        bins = check_df(df_copy, min_r=min_r)
        if len(bins) > 2:
            df = pd.DataFrame(df.copy().apply(lambda x: qq(x, bins)))
            return df  # , len(bins)-1, bins
        else:
            return np.nan  # , len(bins)-1, bins
    else:
        return np.nan  # , len(bins)-1, bins


def categorize(df, quantilize=False, levels=2,
               cols=None, dstart=None, dend=None):
    """
    This function returns the series of categorized columns

    Parameters
    ----------
    df : {Dataframe type}
             The input dataframe

    quantilize : {Boolean type}
                 Default: False

    levels : {Int type}
            Default: 2

    cols : {List type}
            Default: None

    dstart : {Datetime type}
             Default: None

    dend : {Datetime type}
            Default: None

    Return
    ------
    new_df : The output dataframe

       """
    new_df = df.copy()

    cols = get_columns(new_df, cols)
    # #         if dstart is None: dstart=self.index [0]
    # #         if dend is None: dend=self.index [-1]

    if type(levels) == int:
        nlevels = levels
        if nlevels > 0:
            levels = np.array(range(0, 1 + nlevels, 1)) * float(1.0 / nlevels)
    else:
        nlevels = len(levels)

    for col in cols:
        if quantilize:

            subseries = take_columns(new_df, col)
            subseries = take_interval(subseries, dstart=dstart, dend=dend)  # , inplace=True)
            subseries = subseries.dropna()
            if len(subseries) < 10:
                logger.info('!! pas assez de donnée pour la quantilisation pour %s !!')
                return
            levels[0] = -1000000000.0
            levels[-1] = 1000000000.0
            for ilevel in range(1, nlevels):
                levels[ilevel] = subseries.quantile(float(ilevel) / nlevels)
            levels = np.unique(levels)

        catcol = pd.cut(x=new_df[col], bins=levels, labels=False)
        # new_df[new_df[colname] < 0] = np.NaN

    new_df.columns = ['cat']
    return new_df


def take_interval(df, dstart=None, dend=None, inplace=False):

    """
       This function takes a time slice: [dstart, dend]


       Parameters
       ----------
       df : {Dataframe type}
                The input dataframe

       dstart : {Datetime type}
                Default: None

       dend : {Datetime type}
               Default: None

       inplace : {Boolean type}
              Default: False

       Return
       ------
       new_df : The output dataframe

    """

    if len(df.index) == 0:
        return df
    if dstart is None or dstart == '':
        dstart = df.index[0]
    else:
        dstart = pd.to_datetime(dstart)

    if dend is None or dend == '':
        dend = df.index[-1]
    else:
        dend = pd.to_datetime(dend)

    if (dstart <= df.index[0]) and (dend >= df.index[-1]):
        return df

    if inplace:
        # ds=self._as_TDataSet(self [str(dstart) : str(dend)])
        ds = df.loc[df.index <= str(dend)]
        ds = df.loc[df.index >= str(dstart)]
    else:
        # modif a confirmer
        ds = pd.DataFrame(data=df[str(dstart): dend.strftime("%Y-%m-%d")])
    return ds


def calc_modified_duration(df, n, cols=None):
    """
        This function returns the sensitivity series,
        for a series of rates and a maturity

        Parameters
        ----------
        df : {Dataframe type}
                 The input dataframe

        n : {Int type}
            A maturity

        cols : {List type}
                Default: None

        Return
        ------
        new_df : The output dataframe

     """

    cols = take_columns(df, cols)
    cols = np.maximum(cols, 1e-5)
    zc = 1.0 / (1.0 + cols)
    zcn = zc ** n
    res = 1.0
    res -= (n + 1.0) * zcn
    u = (1.0 - zcn)
    u /= (1.0 - zc)
    u *= zc
    res += u
    res /= zc
    res += n * zcn / zc
    res *= - zc * zc

    tabcols = df.columns.values

    res.columns = tabcols
    return df(res)


def time_columns(df):
    """
        This function computes the direct and lagged seasonality variables

        Parameters
        ----------
        df : {Dataframe type}
                 The input dataframe

        Return
        ------
        df : The output dataframe

     """
    df = pd.DataFrame(index=df.index,
                      columns=[glbmetavar + 'DATE', 'MOIS', 'MOIS_', 'JMOIS', 'JMOIS_', 'JSEM', 'JSEM_'])
    df[glbmetavar + 'DATE'] = df.index.year * 10000 + df.index.month * 100 + df.index.day
    df['MOIS'] = df.index.month
    df['MOIS_'] = (df.index.month + 6) % 12
    df['JMOIS'] = df.index.day
    df['JMOIS_'] = (df.index.day + 15) % 31
    df['JSEM'] = (np.rint(df.index.asi8 / glbnano)) % 7
    df['JSEM_'] = (np.rint(df.index.asi8 / glbnano) + 3) % 7
    return df


def apply_filter(df, period=1, min_value=np.NINF, max_value=np.inf, diff_order=1, inpct=True,
                 cols=None):
    """
     This function returns a copy of the current object by canceling
     the abberate values ​​corresponding to returns
     or values ​​outside the min_value and max_value limits

     Parameters
     ----------
     df : {Dataframe type}
           The input dataframe

     period : {Int type}
               Default: 1

     min_value : {Float type}
                 Default: np.NINF

     max_value : {Float type}
                  Default: np.inf

     diff_order : {Int type}
                   Default: 1

     inpct : {Boolean type}
              If True: the series of geometric returns are calculated
              else: the series of arithmetic returns is calculated
              Default: True

     cols : {List type}
            Default: None

     Return
     ------
     new_df : The output dataframe

      """
    df = df.copy()
    cols = get_columns(df, cols)
    datacols = df[cols]

    #  On cherche first_date=première date contenant des données
    first_date = min(datacols.dropna(axis=0, how='any', inplace=False).index)

    #  Et first_value la première valeur associée
    first_value = datacols.loc[first_date][:]

    #  On nettoie ensuite le dataframe datacols
    datacols = datacols[datacols.index != pd.NaT]
    datacols = datacols[datacols.index != pd.NaT]
    datacols.dropna(axis=0, how='all', inplace=True)

    #  On calcule la série des rendements géo si inpct=True
    if diff_order != 0:
        if inpct:
            delta_data = datacols.pct_change(period)

        #  Sinon on calcule la série des rendements arithmétiques
        else:
            delta_data = datacols.diff(period)
    else:
        delta_data = datacols

    #  On retouche les valeurs au-dessus en en-dessous des limites
    if diff_order != 0:
        #  les rendements sont annulés en dehors des limites
        delta_data[delta_data < min_value] = 0
        delta_data[delta_data > max_value] = 0

    else:
        #  Dans le cas de l'ordre 0 on écrase les valeurs
        #  au-dessus des seuils et fait un fwd fill
        delta_data[delta_data < min_value] = np.nan
        delta_data[delta_data > max_value] = np.nan

        #  Il faut traiter le cas particulier où la première valeur
        #  rencontrée dépassait les limites, on fait un np.clip
        #  de cette première valeur que l'on propage en fwd fill
        if (any(np.isnan(delta_data.loc[first_date][:]))):
            fill_value = np.clip(first_value, min_value, max_value)
            delta_data.loc[first_date][:] = fill_value

        delta_data.fillna(method='ffill', inplace=True)

    #  On recontruit les NAVs à partir des rendements
    #  Si on n'est pas à l'ordre 0
    if diff_order != 0:
        if inpct:
            delta_data = np.log(1 + delta_data)

        delta_data = pd.expanding_sum(delta_data)

        #  Dans le cas de rendements géométriques, on recalcule une NAV en normalisant
        #  avec la première valeur valide rencontrée pour la série
        if inpct:
            delta_data = np.exp(delta_data)
            delta_data = delta_data.multiply(first_value, axis=1)

        #  Dans le cas de rendements arithmétiques, la normalisation se fait par addition
        else:
            delta_data = delta_data.add(first_value, axis=1)

    new_df = pd.DataFrame(index=delta_data.index,
                          data=delta_data.values,
                          columns=cols)

    return new_df
