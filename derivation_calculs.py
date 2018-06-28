# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import data_utils

ope_dict = {  # combinaison de deux séries
            'combi': {'shift': 0, 'lag1': 0, 'freq': 'B', 'col1': 0,
                      'coeff1': 1, 'lag2': 0, 'col2': 1, 'coeff2': 0,
                      'lin': True, 'constant': 0, 'mult': 1, 'add': 0,
                      'levels': 0, 'quantilize': True, 'dstart': None,
                      'dend': None, 'catcols': None, 'ownfreq': None,
                      'extend': None},
            # translation des dates de la série sans réalignement:
            # pour changer (retarder) la date de publication des données
            'timeshift': {'shift': 0, 'lag': 0, 'freq': 'B', 'mult': 1,
                          'mult': 1, 'col1': -1, 'add': 0, 'levels': 0,
                          'quantilize': True, 'dstart': None, 'dend': None,
                          'catcols': None, 'ownfreq': None, 'extend': None},
            # décalage dans les valeurs de la série:
            # pour construire une variable décalée
            # Mettre un lag négatif pour prendre une valeur forward
            'lag': {'lag': 1, 'freq': 'B', 'col1': 0, 'add': 0, 'levels': 0,
                    'quantilize': True, 'dstart': None, 'dend': None,
                    'catcols': None, 'ownfreq': None, 'extend': None},
            # variation par rapport aux valeurs passées (period > 0)
            # ou futures (period < 0)
            'delta': {'shift': 0, 'lag': 0, 'period': 1, 'freq': 'B',
                      'mult': 1, 'col1': 0, 'add': 0,
                      'levels': 0, 'quantilize': True,
                      'dstart': None, 'dend': None, 'catcols': None,
                      'ownfreq': None, 'extend': None},

            # variation relative
            'pctdelta': {'shift': 0, 'lag': 0, 'period': 1, 'freq': 'B',
                         'mult': 1, 'col1': 0, 'add': 0, 'ownfreq': None,
                         'extend': None},

            # rendements continus
            'rollingreturn': {'shift': 0, 'lag': 0, 'period': 1,
                              'rollfreq': 'BMS', 'mult': 1, 'add': 0,
                              'col1': 0, 'col2': 1, 'iweek': 1, 'iday': 1,
                              'iroll_interval': 0, 'ownfreq': None,
                              'extend': None},

            # moyenne mobile exponentielle
            'ewma': {'shift': 0, 'lag': 0, 'freq': 'B',
                     'emadecay': 2.0 / (20 + 1), 'wres': True, 'wZ': True,
                     'mult': 1, 'col1': 0, 'add': 0, 'levels': 0,
                     'quantilize': True, 'dstart': None, 'dend': None,
                     'catcols': None, 'ownfreq': None, 'extend': None},

            # volatilité close
            'vol': {'shift': 0, 'lag': 0, 'period': 1, 'freq': 'B',
                    'exponential': True, 'window': 20, 'inpct': True,
                    'annualize': True, 'mult': 1, 'col1': 0, 'add': 0,
                    'levels': 0, 'quantilize': True, 'dstart': None,
                    'dend': None, 'catcols': None,
                    'ownfreq': None, 'extend': None, 'fillinit': True},

            # volatilité OHLC
            'ohlcvol': {'shift': 0, 'lag': 0, 'period': 1, 'freq': 'B',
                        'exponential': True, 'algo': 'yang',
                        'window': 20, 'inpct': True, 'annualize': True,
                        'mult': 1, 'columns': [0, 1, 2, 3], 'add': 0,
                        'levels': 0, 'quantilize': True, 'dstart': None,
                        'dend': None, 'catcols': None,
                        'ownfreq': None, 'extend': None, 'fillinit': True},

            # corrélation
            'corr': {'shift': 0, 'lag1': 0, 'period': 1, 'freq': 'B',
                     'exponential': True, 'window': 20, 'inpct': True,
                     'mult': 1, 'col1': 0, 'col2': 1, 'lag2': 0, 'add': 0,
                     'levels': 0, 'quantilize': True, 'dstart': None,
                     'dend': None, 'catcols': None, 'ownfreq': None,
                     'extend': None, 'fillinit': True},

            # variation de corrélation
            'delta_acorr': {'shift': 0, 'lag1': 0, 'period': 1, 'freq': 'B',
                            'exponential': True, 'window': 20, 'inpct': True,
                            'mult': 1, 'col1': 0, 'col2': 1, 'shortlag': 0,
                            'longlag': 0, 'add': 0, 'levels': 0,
                            'quantilize': True, 'dstart': None,
                            'dend': None, 'catcols': None,
                            'ownfreq': None, 'extend': None, 'fillinit': True},

            # catégorisation
            'cat': {'shift': 0, 'lag': 0, 'quantilize': False,
                    'type_quantilize': '', 'levels': [-1000, 0, 1000],
                    'dstart': None, 'dend': None, 'col1': 0, 'add': 0,
                    'catcols': None, 'ownfreq': None, 'extend': None},

            # sensibilité
            'modifdur': {'maturity': 1},

            # Cumul de rendements
            'cumret': {'shift': 0, 'lag': 0, 'col1': 0, 'add': 0, 'mult': 1,
                       'ownfreq': None, 'extend': None, 'timeweight': False},

            # Remplissage de valeurs manquantes
            'fillmissing': {'main': 1, 'subst': 0, 'add': 0, 'mult': 1},

            # Variables temporelles
            'time': {'shift': 0, 'lag': 0}
            }


def apply_operation(df, freq, operation, parameters, ope_dict, var_list=None):
    operation_dict = ope_dict[operation]

    for key, values in parameters.items():
        operation_dict[key] = values
    if operation == 'timeshift':
        shift = operation_dict['shift']
        mult = operation_dict['mult']
        df_calc = apply_timeshift(df, shift, freq, mult)
        return df_calc
    if operation == 'delta_acorr':
        # df_calc = apply_corr(df)
        df_calc = reduce(lambda x, y: x*y, var_list)
        return df_calc


def apply_timeshift(df, shift, freq, mult):
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
        # Multiplication by mult
        ndf = ndf * mult
        return ndf


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













path = '/home/cluster/git/data_v2/2 Data/1 Received/Market data/Base/FDFD_Index_LAST_PRICE.csv'
var_name = 'FI_STR_USD_1D_LAST'
df= data_utils.load_var(path=path, var_name=var_name)
# print df.head(5)

































