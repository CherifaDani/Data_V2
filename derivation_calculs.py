# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np


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

operation = ''
if operation == 'timeshift':
    