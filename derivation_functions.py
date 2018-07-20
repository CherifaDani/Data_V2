# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
from data_utils import reindex
import math
import logging

from datetime import datetime
from datetime import timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

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


glbNanoSexPerDay = 86400000000000.0
glbFieldSep = '!'
glbMetaVarPrefix = '$$'
glbDefaultTime = datetime(1900, 1, 1, 18, 30)


def extendtodate(df, todate=None, freq='B', limit=5):
    '''Etend un dataset jusqu'à une date plus récente en prolongeant les valeurs. '''
    if todate is None:
        todate = datetime.now()
    else:
        todate = pd.to_datetime(todate)
    dt0 = df.index[-1] + timedelta(days=1)
    if dt0 <= todate:
        dtindex = pd.bdate_range(start=dt0, end=todate, freq=freq)
        newds = pd.DataFrame(index=dtindex, columns=df.columns)
        for col in df.columns:
            newds.ix[0, col] = df.ix[-1, col]
            newds[col] = newds[col].fillna(method='ffill', limit=limit)
        self = df.append(newds)

    return self


def estimate_nat_freq(df, col):
    '''Estime la fréquence naturelle d'une série: la fréquence des changements de valeur '''
    df.dropna()
    df.sort_index(inplace=True)
    # fl = float((df.index.asi8[-1] - df.index.asi8[0]) / glbNanoSexPerDay)
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
    # ddf.deltat[1:] = (ddf.index.asi8[1:] - ddf.index.asi8[: -1]) / glbNanoSexPerDay
    # trier les intervalles entre changements de dates
    lastdelta = ddf.ix[-1]
    ddf.sort(columns='deltat', inplace=True)
    length = len(ddf)
    deltat = ddf.deltat[1:]
    fdict = {}
    # pdb.set_trace()
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
    '''Renvoie des identifiants de colonnes pour un(vecteur de) int ou str. '''
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


def apply_timeshift(df, var_name, shift=1, freq='B', ownfreq=None, refdate=None):
    '''Renvoie une copie de l'objet courant, avec dates translatées d'un délai. '''
    '''Les noms de colonnes de l'objet courant ne sont pas modifiés.'''
    '''freq représente l''unité de compte du décalage'''
    '''ownfreq représente la fréquence finale(propre) de la série.'''
    '''refdate: date de calcul. si fournie, les décalages sont limités à cette date'''
    '''Exemple: décaler de 20j une série trimestrielle'''

    newdataset = df.copy()

    # pas de décalage: on peut changer la fréquence
    # if freq <> 'B':
    if ownfreq is not None and ownfreq != freq:
        pass
        # newdataset=newdataset.change_freq(freq=ownfreq)
    if shift == 0:
        return newdataset

    if refdate is None:
        refdate = datetime.now()
    else:
        refdate = pd.to_datetime(refdate)

    ndf = newdataset.tshift(shift, freq)

    # sous-typer en TDataSet
    # Vérifier que l'on ne décale pas au-delà d'aujourd'hui
    lastdate = ndf.index[-1]
    if lastdate > refdate:
        # lastline=ndf.ix [-1]
        newline = ndf.ix[[-1]]
        ndf = ndf[ndf.index < refdate]
        ndf = ndf.append(newline)

    ndf.columns = [var_name]
    return ndf


def apply_combi(df1, df2, coeff1=1, coeff2=0, constant=0,
                islinear=True, transfo=None):
    '''Renvoie la combinaison linéaire ou exponentielle de deux colonnes. '''
    df1, df2 = reindex(df1, df2)
    # df = pd.concat([df1, df2], axis=1)
    cols1 = get_columns(df1)
    if len(cols1) > 0:
        col1 = cols1[0]
        datacol1 = df1
    else:
        datacol1 = None

    cols2 = get_columns(df2)
    if len(cols2) > 0:
        col2 = cols2[0]
        datacol2 = df2
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
    # [datacol1, datacol2] = reindex(datacol1, datacol2)

    # df = pd.DataFrame()
    if islinear:
        combiarray = np.zeros(len(df1.index)) + constant
        if not c1null:
            combiarray = datacol1.values * coeff1 + constant
        if not c2null:
            combiarray = combiarray + datacol2.values * coeff2

    else:

        # constante égale à 0 en multiplicatif: on la prend pour 1
        if constant == 0:
            constant = 1
        combiarray = np.ones(len(df1.index)) * constant

        if (datacol1 is not None):
            combiarray = np.power(datacol1.values, coeff1) * constant
        if (datacol2 is not None):
            combiarray = combiarray * np.power(datacol2.values, coeff2)

    if transfo is not None:
        if str(transfo).lower() == 'tanh':
            combiarray = np.tanh(combiarray)
        elif str(transfo).lower() == 'sign':
            combiarray = np.sign(combiarray)
    newdataset = pd.DataFrame(index=df1.index, data=combiarray)
    newdataset.columns = ['COMBI']
    return newdataset


def take_columns(df, cols=None, forceuppercase=True):
    '''Equivalent à l'opérateur []'''
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
    # if not(col in self.columns) : return newdataset
    # import pdb; pdb.set_trace()
    # extraction des données à moyenner dans un DataFrame
    datacols = pd.DataFrame(data=df[cols])
    if inplace:
        newdataset = df
    else:
        newdataset = df.copy()
        newdataset = newdataset.take_columns(cols)
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
        newdataset = emadata
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
            newdataset[col] = emadata[col]
            if wres:
                newdataset[col] = rescols[col]
                if normalize:
                    newdataset[col] = zcols[col]

    newdataset.columns = ['EWMA']
    return newdataset


def apply_corr(df1, df2, period=1,
               span=20, exponential=True,
               inpct=True, cols=None, lag=0):
    '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
       period: si 0, corrélation des valeurs, si > 0, corrélation des variations sur period
       lag2: retard sur la seconde colonne
       cols: spécifications de 1 ou 2 colonnes
    '''
    df = pd.concat([df1, df2], axis=1)
    cols = get_columns(df)
    if len(cols) == 1:
        col1 = df1
        col2 = col1
    else:
        col1 = df1
        col2 = df2
    # #  si period == 0 c'est l'autocorrélation des valeurs
    # #  et non des variations qui est calculée
    startval = period + lag * period
    data1 = df1
    data2 = df2
    if period == 0:
        data1 = df1
        data2 = df2.shift(periods=lag)

    else:
        if inpct:
            data1 = data1.pct_change(period)[startval:]

            data2 = data2.pct_change(period).shift(periods=lag * period)[startval:]

        else:
            data1 = data1.diff(period)[startval:]
            data2 = data2.diff(period).shift(periods=lag * period)[startval:]
    # if exponential:
    corrdata = pd.ewmcorr(data1[startval:], data2[startval:], span=span)
    # else:
    #     corrdata = pd.rolling_corr(data1, data2, window=span)
    # pdb.set_trace()
    # voldata=pd.rolling_mean(diffdata, window=window)+

    # newdataset =
    # corrname = self.name + glbFieldSep + 'CORR'
    # corrname = corrname + '[' + str(col1) + ',' + str(col2) + ',' + str(span) + ']'
    # newdataset = corrdata.dropna()
    # # newdataset['CORR'] = corrdata
    # # nom de la colonne: radical + Vol
    # # newcols [icol]=col.split(fieldsep)[0] + fieldsep + 'VOL@'+ str(window)
    # # newdataset.name = corrname
    # newdataset.columns = ['CORR']
    # return newdataset
    return corrdata
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
#     newdataset = pd.DataFrame(index=dfx.index)
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
#     # newdataset['CORR'] = corrdata
#     newdataset = corrdata
#     newdataset.columns = ['CORR']
#
#     # print(newdataset)
#     return newdataset


def apply_vol(df,
              period=1,
              window=20, inplace=True,
              annualize=True,
              fillinit=True,
              inpct=True,
              cols=None
              ):
    '''Renvoie la série des volatilités de rendements '''
    if inplace:
        newdataset = df
    else:
        newdataset = df.copy()
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
        newdataset[col] = voldata[voldata.columns[icol]] * annfactor

    # newdataset.columns = newcols
    newdataset.columns = ['VOL']
    print(newdataset)
    return newdataset


def take_diff(df, period=1, inplace=False, cols=None, inpct=True,
              fieldsep='', alldays=True, ownfreq=None):
    '''Renvoie la série des différences d'une colonne pour un décalage donné.
       En vue d'une synchronisation ultérieure dans une matrice, il faut pré-remplir les différences
       par des zéros à toutes les dates ne figurant pas dans l'index.
       # CG 14/6/2: introduction de l'argument ownfreq représentant la fréquence naturelle de la série
       Celle-ci est nécessaire dans le cas d'une série mensuelle présentée quotidiennement,
       avec donc un saut par mois.
    '''

    if inplace:
        newdataset = df
    else:
        newdataset = None
    if fieldsep == '':
        fieldsep = glbFieldSep
    cols = get_columns(df, cols)
    # if not(col in self.columns) : return newdataset
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
            newdataset[col] = deltadata[col]
    else:
        # si l'objet diffseries est un pd.Series, il n'a pas de champ columns
        newdataset = pd.DataFrame(index=deltadata.index,
                                  data=deltadata.values,
                                  columns=cols)
    # newdataset.columns = newcols
    # newdataset = newdataset.asfreq(freq='B')
    # newdataset.reindex(index=idx_all)
    newdataset.columns = ['X']
    return newdataset


def apply_rolling(maincol, substcol,  rollfreq, iday, iweek, effectiveroll_lag=0, inpct=True):
    '''
    Renvoie la série des variations d'une colonne pour un décalage donné.
    Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
    Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
    où Vsubst représente la série self [substcol]
    '''

    assert type(rollfreq) == str
    assert iday >= 0
    assert iweek >= 0
    period = 1
    assert effectiveroll_lag in [0, 1]
    df = pd.concat([maincol, substcol], axis=1)
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
    newdataset = pd.DataFrame(index=retdata.index,
                              data=retdata.values,
                              columns=[newcol])
    # revenir au calendrier restreint
    newdataset = pd.DataFrame(newdataset.dropna())
    newdataset.columns = [newcol]
    # newdataset.name = retdataname
    return newdataset


def set_daytime(df, hm_time, dates=None):
    """
    Fixe l'heure de la journée pour tout l'index ou pour des dates données.

    hm_time: datetime ou Timestamp, heure/minute/seconde à appliquer aux dates
    dates: liste de dates dont on veut modif ier l'heure
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


def apply_lag(df, lag=1, freq=None, cols=None, inplace=False):
    '''Renvoie une copie de l'objet courant, avec valeurs décalées d'un retard et réalignées. '''
    '''Les noms de colonnes de l'objet courant ne sont pas modifiés.'''
    newdataset = df.copy()
    if lag == 0:
        return newdataset

    cols = get_columns(df, cols)
    # if not(cols in self.columns) : return newdataset
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
            newdataset[col] = laggedseries[col]
    else:
        newdataset = pd.DataFrame(index=laggedseries.index,
                                  data=laggedseries.values,
                                  columns=cols)

    # idx_all = pd.bdate_range(start=df.index[0], end=df.index[-1], freq=freq)
    # newdataset = newdataset.reindex(index=idx_all, method=None)
    # bday_us = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # newdataset = newdataset.resample(bday_us)
    # # print('newda')
    # print(newdataset)
    # newdataset.to_csv('vv.csv')
    return newdataset


def apply_cumulative_return(df, timeweight=False, cols=None, inplace=True):
    '''Renvoie le cumul composé des rendements'''
    '''AMBIGU quand inplace=True, cols <> None'''
    cols = get_columns(df, cols)
    # pdb.set_trace()
    # datacols=pd.DataFrame(data=self [cols])
    if timeweight is True:
        deltatime = pd.Series(df.index.asi8)
        deltatime = deltatime.diff(1) / glbNanoSexPerDay
        deltatime.fillna(value=0.0, inplace=True)
        deltatime = deltatime / 365.25
        deltatime = deltatime.reshape(len(df.index), 1)
        df[cols] = df[cols] * deltatime
    navdata = np.log(1 + df[cols])
    navdata = pd.expanding_sum(navdata)
    navdata = np.exp(navdata)
    newcols = range(len(cols))

    if inplace:
        newdataset = df
    else:
        newdataset = pd.DataFrame(data=navdata)
    for icol, col in enumerate(cols):
        if inplace:
            newdataset[col] = navdata[col]

    newdataset.columns = newcols

    return newdataset


def apply_futures_roll(col_c1, col_c2, roll_dict):
    """
            roll_futures:
            Calculates roll_adjusted returns on a series of first and second futures, for a rolling rule

            df: dataframe containing two columns of closing prices for c1 and c2
            roll_dict: dictionary describing the rolling rule, like in apply_roll_shift
            col_c1: index of column storing contract 1 prices
            col_c2: index of column storing contract 2 prices

            Returns:
            a dataframe with the following columns:
            C1: the price of c1 contract
            C1_ROLL: the synthetic price of roll-adjusted c1 contract
            RETURN_1D_AFTER_ROLL: the daily roll-adjusted return
            """
    df = pd.concat([col_c1, col_c2], axis=1)
    df.dropna(inplace=True)
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
    apply_roll_shift:
    returns a series of rolling dates from an initial list of reference dates and a dictionary

    dates: a list of dates
    roll_dict: a dictionary with the following entries
        'freq': the frequency of rolling dates, default 'BQ'
        'day': if specified, the hard-coded calendar day in month of the rolling date, default -1(not specified)
        'week': if day==-1, the index of week roll in month(starting at 0), default 0
        'weekday': if day==-1, the index of day in week(starting at 0 for Mondays)
        'bday_offset': an extra shift in business days, generally negative or zero, default 0
        'bmonth_offset' an extra shift in months, generally negative or zero, default 0

    RETURNS:
    The list of last rolling dates preceding each element of the given reference dates
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
    df_rolldates = pd.DataFrame(index=b_enddates, columns=['LAST_TRADING_DATE'])
    df_rolldates['LAST_TRADING_DATE'] = b_enddates
    if roll_dict['bmonth_offset'] != 0:
        df_rolldates['LAST_TRADING_DATE'] = map(lambda d: monthrollrule.rollback(d),
                                                df_rolldates['LAST_TRADING_DATE'])
    if roll_day >= 0:
        df_rolldates['LAST_TRADING_DATE'] = map(lambda d: d.replace(day=roll_day),
                                                df_rolldates['LAST_TRADING_DATE'])
    else:
        df_rolldates['LAST_TRADING_DATE'] = map(lambda d: rollrule.rollback(d),
                                                df_rolldates['LAST_TRADING_DATE'])

    if roll_dict['bday_offset'] != 0:
        df_rolldates['LAST_TRADING_DATE'] = map(lambda d: d + bdayrollrule,
                                                df_rolldates['LAST_TRADING_DATE'])
    df_rolldates.dropna(inplace=True)
    return df_rolldates


def fill_missing_values(df1, df2, idxmain=0, idxsubst=1, dfsubst=None):
    '''Remplit les valeurs manquantes de la colonne idxmain par la colonne idxsubst '''
    df = pd.concat([df1, df2], axis=1)
    if dfsubst is None:
        df2 = df
    else:
        df2 = dfsubst
    try:
        maincol = get_columns(idxmain)[0]
        substcol = get_columns(idxsubst)[0]
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



# def apply_ohlc_vol(df,
#                    window=20,
#                    annualize=True,
#                    fillinit=True,
#                    inpct=True,
#                    ohlccols=None,
#                    algo='yang'
#                    ):
#     '''Renvoie la série des volatilités de rendements '''
#
#     cols = get_columns(df, ohlccols)
#     if len(cols) < 4:
#         return None
#
#     ocol = cols[0]
#     hcol = cols[1]
#     lcol = cols[2]
#     ccol = cols[3]
#     if inpct:
#         oreturn = self[ocol] / self[ccol].shift(1) - 1
#         creturn = self[ccol] / self[ocol] - 1
#         hreturn = self[hcol] / self[ocol] - 1
#         lreturn = self[lcol] / self[ocol] - 1
#     else:
#         oreturn = self[ocol] - self[ccol].shift(1)
#         creturn = self[ccol] - self[ocol]
#         hreturn = self[hcol] - self[ocol]
#         lreturn = self[lcol] - self[ocol]
#
#     ovar = pd.rolling_var(oreturn, window=window)
#     closevar = pd.rolling_var(oreturn + creturn, window=window)
#
#     if algo == 'park':
#         lhvar = pd.rolling_var(hreturn - lreturn, window=window)
#         retvar = lhvar / (4 * math.log(2))
#     else:
#         sqret = hreturn * (hreturn - creturn) + lreturn * (lreturn - creturn)
#         if algo == 'rogers':
#             retvar = pd.rolling_mean(sqret, window=window)
#         elif algo == 'yang':
#             k = 0.0783
#             sqret = hreturn * (hreturn - creturn) + lreturn * (lreturn - creturn)
#             retvar = ovar + k * closevar + (1 - k) * pd.rolling_mean(sqret, window=window)
#
#     voldata = np.sqrt(retvar)
#     if fillinit:
#         voldata[0: window + 1] = voldata[0: window + 1].fillna(method='bfill')
#     voldata = voldata.dropna()
#     annfactor = 1
#     # pdb.set_trace()
#     # voldata=pd.rolling_mean(diffdata, window=window)
#     volname = self.name + glbFieldSep + 'VOL' + algo[0] + '@' + str(window)
#     if annualize:
#         nfreqdict = self.estimate_nat_freq(cols[0])
#         nfreq = max(1, nfreqdict['min'])
#         annfactor = math.sqrt(260 / nfreq)
#     else:
#         annfactor = 1
#
#     newdataset = TDataSet(voldata * annfactor)
#     # nom de la colonne: radical + Vol
#     newdataset.name = volname
#     newdataset.columns = [volname]
#     return newdataset
#
# def apply_timeshift(df, freq, shift=0):
#         """
#         Renvoie une copie de l'objet courant, avec dates translatées
#         d'un délai.
#         Les noms de colonnes de l'objet courant ne sont pas modifiés.
#         freq représente l''unité de compte du décalage
#         ownfreq représente la fréquence finale(propre) de la série.
#         refdate: date de calcul. si fournie, les décalages sont limités
#         à cette date
#         Exemple: décaler de 20j une série trimestrielle
#         """
#         # Shiffting with a given shift
#
#         ndf = df.tshift(shift, freq)
#         return ndf
#
#
# def apply_combi(df1, df2, coeff1, coeff2, islinear, transfo=None):
#     dfs = pd.DataFrame()
#     dfx, dfy = reindex(df1, df2)
#     if islinear:
#         dfs = coeff1 * dfx + coeff2 * dfy
#     else:
#         dfs = (coeff1 ** dfx) * (coeff2 ** dfy)
#
#     if transfo is not None:
#         if str(transfo).lower() == 'tanh':
#             transfo_df = np.tanh(dfs)
#         elif str(transfo).lower() == 'sign':
#             transfo_df = np.sign(dfs)
#         return transfo_df
#     else:
#         return dfs
#
#
# def apply_corr(df, period=1, exponential=True, inpct=True, lag=0, span=20):
#     """
#
#     :param df:
#     :param period:
#     :param span:
#     :param exponential:
#     :param inpct:
#     :param lag:
#     :return:
#     '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
#            period: si 0, corrélation des valeurs, si > 0, corrélation des
#            variations sur period
#            lag: retard sur la seconde colonne
#            cols: spécifications de 1 ou 2 colonnes
#     rolling_corr(data1, data2, window=span)
# """
#
#     # #  si period == 0 c'est l'autocorrélation des valeurs
#     # #  et non des variations qui est calculée
#     startval = period + lag * period
#     if period == 0:
#         data1 = df
#         data2 = df.shift(periods=lag)
#     else:
#         if inpct:
#             data1 = df.pct_change(period)[startval:]
#             data2 = df.pct_change(period).shift(periods=lag * period)[startval:]
#         else:
#             data1 = df.diff(period)[startval:]
#             data2 = df.diff(period).shift(periods=lag * period)[startval:]
#
#     if exponential:
#         corrdata = pd.ewmcorr(data1[startval:], data2[startval:], span=span)
#     else:
#         corrdata = pd.rolling_corr(data1, data2, window=span)
#
#     corrdata = corrdata.dropna()
#     return corrdata
#
#
# def apply_pctdelta(df, period, freq, inpct):
#
#     if inpct:
#         deltadata = df.pct_change(period)
#     else:
#         deltadata = df.diff(period)
#
#     idx_all = pd.bdate_range(start=(deltadata.index[0]).date(),
#                              end=(deltadata.index[-1]).date(),
#                              freq=freq)
#
#     # Reindex using datetime index, to drop hours and minutes
#     deltadata.index = pd.DatetimeIndex(deltadata.index).normalize()
#     if(freq == 'B' or freq == 'D'):
#         deltadata = deltadata.reindex(index=idx_all, method=None)
#
#     else:
#         deltadata = deltadata.reindex(index=idx_all, method='pad')
#
#     return deltadata
#
#
# def apply_rolling(df1, df2, rollfreq, iweek, iday, iroll_interval, freq):
#     """
#         Renvoie la série des variations d'une colonne pour un décalage donné.
#         Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
#         Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
#         où Vsubst représente la série self [substcol]
#     """
#     # élargir le calendrier pour inclure les dates de rolls de façon certaine
#     idx_all = pd.bdate_range(start=(df1.index[0]).date(),
#                              end=(df1.index[-1]).date(),
#                              freq=freq)
#     df1.index = pd.DatetimeIndex(df1.index).normalize()
#     data = df1.reindex(index=idx_all, method=None)
#     rolldates = pd.bdate_range(data.index[0], data.index[-1], freq=rollfreq)
#     rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)
#
#
# def apply_ewma(df, emadecay, wres, wz):
#     """
#     Renvoie la série des ema d un ensemble de colonnes pour une pseudo durée(span) donnée
#      self: contient la totalité des données primaires dont on veut calculer la moyenne
#     emadecay: coefficient d'atténuation de la moyenne(proche de 1). Prioritaire si fourni.
#     span: 2/emadecay - 1
#     cols: groupe de colonnes dont on calcule l'ewma.
#     wres: si True, on calcule également le résidu
#     normalize: si True, on calcule aussi le Z-Score(résidu / ewmastd(même span))
#     histoemadata: série facultative contenant les valeurs historiques de l'ewma sur des dates
#        normalement antérieures aux données primaires.
#     overridedepth: nombre de jours passés(à partir de la donnée la plus récente) à recalculer
#     """
#     stdev_min = 1e-5
#     rescols = pd.DataFrame()
#     zcols = pd.DataFrame()
#     # calculer la période synthétique correspondant au coeff s'il est fourni
#     if type(emadecay) in [int, float]:
#         if emadecay > 0:
#             span = (2.0 / emadecay) - 1
#     df_calc = pd.ewma(df, span=span, adjust=True)
#     # calcul du résidu
#     if wres:
#         rescols = df - df_calc
#         # calcul du ZScore
#         if wz:
#             stdevcols = pd.ewmstd(rescols, span=span)
#             stdevcols[stdevcols <= stdev_min] = np.nan
#             zcols = rescols * 0.0
#             zcols[stdevcols > 0] = rescols[stdevcols > 0] / stdevcols[stdevcols > 0]
#
#     if not wres:
#         return df_calc
#     elif wz is True:
#         return  zcols
#     else:
#         return  rescols
#
#
# def apply_fill(df1, df2, dfsubst):
#
#     return None
#
#
# def apply_roll_shift(dates, roll_dict):
#     """
#     apply_roll_shift:
#     returns a series of rolling dates from an initial list of reference dates and a dictionary
#
#     dates: a list of dates
#     roll_dict: a dictionary with the following entries
#         'freq': the frequency of rolling dates, default 'BQ'
#         'day': if specified, the hard-coded calendar day in month of the rolling date, default -1(not specified)
#         'week': if day==-1, the index of week roll in month(starting at 0), default 0
#         'weekday': if day==-1, the index of day in week(starting at 0 for Mondays)
#         'bday_offset': an extra shift in business days, generally negative or zero, default 0
#         'bmonth_offset' an extra shift in months, generally negative or zero, default 0
#
#     RETURNS:
#     The list of last rolling dates preceding each element of the given reference dates
#     """
#     if type(dates) is not list:
#         dates = [dates]
#
#     roll_freq = roll_dict['freq']
#     # les dates de fin de périodes(mois ou trimestre ouvré)
#     b_enddates = pd.bdate_range(dates[0], dates[-1], freq=roll_freq)
#     b_enddates = map(lambda d: d.replace(hour=dates[0].hour,
#                                          minute=dates[0].minute,
#                                          second=dates[0].second), b_enddates)
#     rollrule = pd.datetools.WeekOfMonth(weekday=roll_dict['weekday'],
#                                         week=roll_dict['week'])
#     bdayrollrule = pd.datetools.BDay(n=roll_dict['bday_offset'])
#     monthrollrule = pd.datetools.BMonthEnd(n=roll_dict['bmonth_offset'])
#     roll_day = roll_dict['day']
#
#     # les dates de roll
#     df_rolldates = pd.DataFrame(index=b_enddates, columns=['LAST_TRADING_DATE'])
#     df_rolldates['LAST_TRADING_DATE'] = b_enddates
#     if roll_dict['bmonth_offset'] != 0:
#         df_rolldates['LAST_TRADING_DATE'] = map(lambda d: monthrollrule.rollback(d),
#                                                 df_rolldates['LAST_TRADING_DATE'])
#     if roll_day >= 0:
#         df_rolldates['LAST_TRADING_DATE'] = map(lambda d: d.replace(day=roll_day),
#                                                 df_rolldates['LAST_TRADING_DATE'])
#     else:
#         df_rolldates['LAST_TRADING_DATE'] = map(lambda d: rollrule.rollback(d),
#                                                 df_rolldates['LAST_TRADING_DATE'])
#
#     if roll_dict['bday_offset'] != 0:
#         df_rolldates['LAST_TRADING_DATE'] = map(lambda d: d + bdayrollrule,
#                                                 df_rolldates['LAST_TRADING_DATE'])
#     df_rolldates.dropna(inplace=True)
#     return df_rolldates
#
#
# def apply_futures_roll(df1, df2, rolldict):
#     """
#     roll_futures:
#     Calculates roll_adjusted returns on a series of first and second futures, for a rolling rule
#
#     df: dataframe containing two columns of closing prices for c1 and c2
#     roll_dict: dictionary describing the rolling rule, like in apply_roll_shift
#     col_c1: index of column storing contract 1 prices
#     col_c2: index of column storing contract 2 prices
#
#     Returns:
#     a dataframe with the following columns:
#     C1: the price of c1 contract
#     C1_ROLL: the synthetic price of roll-adjusted c1 contract
#     RETURN_1D_AFTER_ROLL: the daily roll-adjusted return
#     """
#     df = pd.concat([df1, df2], axis=1)
#     df.dropna(inplace=True)
#     h_c1, m_c1, s_c1 = df.index[0].hour, df.index[0].minute, df.index[0].second
#     dstart = df.index[0]
#     dend = df.index[-1]
#     bdates = pd.bdate_range(dstart, dend, freq='B')
#     bdates = map(lambda d: d.replace(hour=h_c1, minute=m_c1, second=s_c1), bdates)
#     # les dates de roll
#     df_rolldates = apply_roll_shift(bdates, rolldict)
#     # les rendements quotidiens
#     df1 = df[df.columns[0]]
#     df2 = df[df.columns[1]]
#
#     df_ret1 = df1.dropna().pct_change(periods=1)
#     df_ret1.columns = ['RETURN_1D']
#     # rendements modifiés après prise en compte du roll
#     # df_ret1['RETURN_1D_AFTER_ROLL'] = df_ret1['RETURN_1D']
#     # le jour suivant le roll dans le calendrier du contrat
#     df_ret1['NEXT_BDATE'] = np.nan
#     df_ret1['NEXT_BDATE'][:-1] = df_ret1.index[1:]
#     # les cours des 2 premiers contrats
#     df_ret1['C1'] = df1.dropna()
#     df_ret1['C2'] = df2.dropna()
#     # # le ratio c1 / c2 prolongé à tous les jours cotés
#     # df_ret1['RATIO12'] = (df_ret1['C1'] / df_ret1['C2']).fillna(method='ffill')
#     # next_bd_after_roll = df_ret1.loc[df_rolldates['LAST_TRADING_DATE'],
#     #                                  'NEXT_BDATE'].fillna(method='ffill').fillna(method='bfill')
#     # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = 1.0 + df_ret1.loc[
#     #     next_bd_after_roll, 'RETURN_1D_AFTER_ROLL']
#     # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] *= df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] * \
#     #                                                            df_ret1.loc[
#     #                                                                df_rolldates['LAST_TRADING_DATE'], 'RATIO12'].values
#     # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = df_ret1.loc[
#     #                                                               next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] - 1.0
#     #
#     # df_roll = pd.DataFrame(index=df_ret1.index, columns=['C1', 'C1_ROLL', 'RETURN_1D_AFTER_ROLL', 'RETURN_1D', 'C2'])
#     # df_roll['C1_ROLL'] = 1.0
#     #
#     # roll_ret = np.log(1 + df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL'])
#     # roll_ret = pd.expanding_sum(roll_ret)
#     # roll_nav = np.exp(roll_ret)
#     # df_roll['C1_ROLL'] = roll_nav * df.iloc[0, 0]
#     # df_roll['C1'] = df.take_columns(col_c1)
#     # df_roll['RETURN_1D_AFTER_ROLL'] = df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL']
#     # df_roll['RETURN_1D'] = df_ret1.loc[:, 'RETURN_1D']
#     # df_roll['C2'] = df_ret1.loc[:, 'C2']
#     return df_ret1
#
#
# def estimate_nat_freq(df):
#     '''Estime la fréquence naturelle d'une série: la fréquence des changements de valeur '''
#     df.dropna()
#     df.sort_index(inplace=True)
#     fl = float((df.index.asi8[-1] - df.index.asi8[0]) / glbnano)
#
#     # série des différences
#     ddf = df.diff(1)
#     # série des différences non nulles
#     ddf = df[ddf != 0]
#     # rajouter une colonne pour les différences de dates
#     ddf['deltat'] = 0
#     ddf.deltat[1:] = (ddf.index.asi8[1:] - ddf.index.asi8[: -1]) / glbnano
#     # trier les intervalles entre changements de dates
#     lastdelta = ddf.ix[-1]
#     ddf.sort(columns='deltat', inplace=True)
#     l = len(ddf)
#     deltat = ddf.deltat[1:]
#     fdict = {}
#
#     if l > 1:
#         fdict['last'] = lastdelta
#         fdict['min'] = mind = deltat.min()
#         fdict['datemin'] = deltat.idxmin()
#         fdict['pct5'] = mind
#         fdict['pct10'] = mind
#         fdict['pct25'] = mind
#         fdict['median'] = deltat.ix[int(0.5 * l) - 1]
#         fdict['max'] = maxd = deltat.max()
#         fdict['datemax'] = deltat.idxmax()
#         fdict['pct95'] = maxd
#         fdict['pct90'] = maxd
#         fdict['pct75'] = maxd
#         fdict['n1'] = len(deltat[deltat >= 1])
#         fdict['r1'] = fdict['n1'] / fl
#         fdict['n5'] = len(deltat[deltat >= 5])
#         fdict['r5'] = fdict['n5'] / (fl / 5)
#         fdict['n10'] = len(deltat[deltat >= 10])
#         fdict['r10'] = fdict['n10'] / (fl / 10)
#         fdict['n20'] = len(deltat[deltat >= 20])
#         fdict['r20'] = fdict['n20'] / (fl / 20)
#         if l > 4:
#             fdict['pct25'] = deltat.ix[int(0.25 * l) - 1]
#             fdict['pct75'] = deltat.ix[int(0.75 * l) - 1]
#             if l > 10:
#                 fdict['pct10'] = deltat.ix[int(0.1 * l) - 1]
#                 fdict['pct90'] = deltat.ix[int(0.9 * l) - 1]
#                 if l > 20:
#                     fdict['pct5'] = deltat.ix[int(0.05 * l) - 1]
#                     fdict['pct95'] = deltat.ix[int(0.95 * l) - 1]
#     return fdict
#
#
# def apply_vol(df, period, window, inpct, annualize, fillinit, freq):
#     '''Renvoie la série des volatilités de rendements '''
#     if period != 0:
#         diffdata = apply_pctdelta(df, period, freq, inpct)
#     else:
#         diffdata = df
#     voldata = pd.rolling_std(diffdata, window=window)
#     if fillinit:
#         voldata[0: window] = voldata[0: window + 1].fillna(method='bfill')
#     voldata = voldata.dropna()
#     cols = df.columns
#     newcols = range(len(cols))
#     for icol, col in enumerate(cols):
#         if annualize:
#             nfreqdict = estimate_nat_freq(col)
#             nfreq = max(1, nfreqdict['min'])
#             annfactor = math.sqrt(260 / nfreq)
#         else:
#             annfactor = 1
#         newdf = voldata[voldata.columns[icol]] * annfactor
#
#     return newdf
#
#
# def apply_ohlcvol(df):
#     pass
#
#
# def apply_cumulative_return(df, timeweight=False):
#     '''Renvoie le cumul composé des rendements'''
#     '''AMBIGU quand inplace=True, cols <> None'''
#     if timeweight is True:
#         deltatime = pd.Series(df.index.asi8)
#         deltatime = deltatime.diff(1) / glbnano
#         deltatime.fillna(value=0.0, inplace=True)
#         deltatime = deltatime / 365.25
#         deltatime = deltatime.reshape(len(df.index), 1)
#         df = df * deltatime
#     navdata = np.log(1 + df)
#     navdata = pd.expanding_sum(navdata)
#     navdata = np.exp(navdata)
#     return navdata

# def apply_rolling(df1, df2,  rollfreq, iday, iweek, effectiveroll_lag=0, inpct=True):
#     '''
#     Renvoie la série des variations d'une colonne pour un décalage donné.
#     Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
#     Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
#     où Vsubst représente la série self [substcol]
#     '''
#
#     assert type(rollfreq) == str
#     assert iday >= 0
#     assert iweek >= 0
#     period = 1
#     # assert type(period) == int
#     # assert period > 0
#     assert effectiveroll_lag in [0, 1]
#
#     # cols1 = get_columns(df1)
#     # cols2 = get_columns(df2)
#     # cols = [cols1, cols2]
#     # if cols is None:
#     #     return None
#     dfx, dfy = reindex(df1, df2)
#     df = pd.concat([dfx, dfy], axis=1)
#     # maincol = cols[0]
#     # substcol = cols[1]
#     # datacols = df[maincol]
#
#     if not inpct:
#         retdata = dfx.diff(period)
#     else:
#         retdata = dfx.pct_change(period)
#     idx_all = pd.bdate_range(start=dfx.index[0], end=dfx.index[-1], freq='B')
#     # effacer l'heure pour synchronisation
#     retdata = set_daytime(retdata, datetime(2000, 1, 1))
#     set_daytime(dfx, datetime(2000, 1, 1))
#     # élargir le calendrier pour inclure les dates de rolls de façon certaine
#     retdata = retdata.reindex(index=idx_all, method=None)
#
#     # générer la série des dates de roll
#     #          if rollfreq [1:].find('BS') < 0:
#     #              rollfreq=rollfreq + 'BS'
#     rolldates = pd.bdate_range(start=dfx.index[0], end=dfx.index[-1], freq=rollfreq)
#     rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)
#     # Ne garder que les dates de roll antérieures aux données courantes
#     rolldates = rolldates[rolldates <= retdata.index[-1]]
#     daybefore_rolldates = rolldates + pd.datetools.BDay(-period)
#     dayafter_rolldates = rolldates + pd.datetools.BDay(period)
#
#     # timeidx=self.index
#     # Contrat M(front) coté jusqu'à TRoll, traité jusqu'en TRoll-1, roulé en TRoll-1
#     # Contrat M+1(next), coté jusqu'à TRoll, devient le front en TRoll + 1, traité en TRoll-1
#     # Returns:
#     #  en TRoll, Close(F2, TRoll)/ Close(F2, TRoll-1) - 1
#     #  en TRoll + 1, Close(F1, TRoll+1)/ Close(F2, TRoll-1) - 1
#     # dayafter_roll_contract=maincol
#     # cas de UX
#     if effectiveroll_lag == 0:
#         roll_contract = dfx
#         daybefore_roll_contract = dfx
#     # cas de FVS
#     else:
#         roll_contract = dfy
#         daybefore_roll_contract = dfy
#
#     if inpct:
#         rollday_returns = dfx.loc[rolldates, roll_contract].values / \
#                           dfx.loc[daybefore_rolldates, daybefore_roll_contract].values - 1
#         dayafter_returns = dfx.loc[dayafter_rolldates, dfx].values / \
#                            dfx.loc[rolldates, dfy].values - 1
#     else:
#         rollday_returns = dfx.loc[rolldates, roll_contract].values - \
#                           dfx.loc[daybefore_rolldates, daybefore_roll_contract].values
#         dayafter_returns = dfx.loc[dayafter_rolldates, dfx].values - \
#                            dfx.loc[rolldates, dfy].values
#
#     retdata.loc[rolldates, dfx] = rollday_returns
#     retdata.loc[dayafter_rolldates, dfx] = dayafter_returns
#
#     newdataset = pd.DataFrame(index=retdata.index,
#                               data=retdata.values,
#                               columns=var_name)
#     # revenir au calendrier restreint
#     newdataset = newdataset.dropna()
#     return newdataset
