# -*- coding: utf-8 -*-
import pandas as pd
import derivation_functions as dfunc

glbnano = 86400000000000.0


def read_df(x):
    return x.read_var(x.get_param('path'))


def get_var_name(x):
    return x.get_param('var_name')


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
    var_names = map(lambda x: get_var_name(x), var_list)
    idx0 = dfs[0]
    # var_name0 = var_names[0]
    idx1 = dfs[1] if len(list(dfs)) == 2 else None
    # var_name1 = var_names[1] if len(list(var_names)) == 2 else None
    col1 = parameters.get('col1', 0)
    if col1 == 1:
        idx0 = idx1
        idx1 = idx0
    # if 'shift' in parameters:
    #     # décalage des dates pour tenir compte de la disponibilité réelle des données
    #     # s'applique à toutes les colonnes d'un dataset
    #     shift = parameters['shift']
    #     if shift != 0 or freq is not None:
    #         shifteddata = dfunc.apply_timeshift(idx0, shift=shift, freq=freq)
    if operation == 'timeshift':
        shift = parameters.get('shift', 0)
        fdf = dfunc.apply_timeshift(idx0, freq, shift)

    elif operation == 'fillmissing':
        idxmain = parameters.get('main', 0)
        idxsubst = parameters.get('subst', 1)
        fdf = dfunc.fill_missing_values(idx0, idx1, idxmain=idxmain, idxsubst=idxsubst)

    elif operation == 'combi':
        coeff1 = parameters.get('coeff1', 1)
        coeff2 = parameters.get('coeff2', 0)
        islinear = parameters.get('lin', True)
        transfo = parameters.get('transfo', None)
        fdf = dfunc.apply_combi(idx0, idx1, coeff1=coeff1, coeff2=coeff2, islinear=islinear, transfo=transfo)

    elif operation == 'pctdelta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        fdf = dfunc.take_diff(idx0, period=period, ownfreq=ownfreq, inpct=True)

    elif operation == 'delta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        fdf = dfunc.take_diff(df=idx0, period=period, inplace=False,
                             inpct=False,
                             ownfreq=freq)


    elif operation == 'rollingreturn':
        period = parameters.get('period', 1)
        rollfreq = parameters.get('rollfreq', 'B')
        iweek = parameters.get('iweek', 1)
        iday = parameters.get('iday', 1)
        iroll_interval = parameters.get('iroll_interval', 0)
        fdf = dfunc.apply_rolling(maincol=idx0, substcol=idx1, rollfreq=rollfreq,
                                  iweek=iweek, iday=iday, effectiveroll_lag=iroll_interval,
                                  inpct=True)

    elif operation == 'ewma':
        emadecay = parameters.get('emadecay', 2.0 / (20 + 1))
        wres = parameters.get('wres', True)
        # wz = parameters.get('wZ', True)
        fdf = dfunc.apply_ewma(df=idx0, emadecay=emadecay, wres=wres, inplace=True, normalize=True, stdev_min=1e-5, histoemadata=None)

    elif operation == 'futuresroll':
        rolldict = {'freq': parameters.get('freq', 'B'),
                    'week': parameters.get('week', 0),
                    'weekday': parameters.get('weekday', 0),
                    'day': parameters.get('day', -1),
                    'bday_offset': parameters.get('bday_offset', 0),
                    'bmonth_offset': parameters.get('bmonth_offset', 0)
                    }
        fdf = dfunc.apply_futures_roll(col_c1=idx0, col_c2=idx1, roll_dict=rolldict)

    elif operation == 'vol':
        period = parameters.get('period', 1)
        window = parameters.get('window', 20)
        inpct = parameters.get('inpct', True)
        annualize = parameters.get('annualize', True)
        fillinit = parameters.get('fillinit', True)
        fdf = dfunc.apply_vol(df=idx0, period=period, window=window, inpct=inpct, annualize=annualize, fillinit=fillinit)

    elif operation == 'ohlcvol':
        period = parameters['period']
        window = parameters['window']
        inpct = parameters['inpct']
        annualize = parameters['annualize']
        fillinit = parameters['fillinit']
        algo = parameters['algo']
        columns = parameters['columns']
        fdf = dfunc.apply_ohlc_vol(df=idx0, OHLCcols=columns,
                             window=window, inpct=inpct,
                             annualize=annualize,
                             fillinit=fillinit,
                             algo=algo)
    elif operation == 'corr':
        period = parameters['period']
        window = parameters['window']
        inpct = parameters['inpct']
        exponential = parameters['exponential']
        lag = parameters['lag2']
        fdf = dfunc.apply_corr(df1=idx0, df2=idx1, period=period, inpct=inpct,
                               lag=lag,
                               exponential=exponential, span=window, cols=None)
    elif operation == 'delta_acorr':
        period = parameters.get('period', 0)
        shortwindow = parameters.get('shortwindow', 20)
        longwindow = parameters.get('longwindow', 100)
        lag = parameters.get('lag', 0)
        inpct = parameters.get('inpct', True)
        exponential = parameters.get('exponential', True)
        acorrshort = dfunc.apply_corr(df1=idx0, df2=idx0, period=period, inpct=inpct, lag=lag,
                                      exponential=exponential, span=shortwindow, cols=None)
        acorrlong = dfunc.apply_corr(df1=idx0, df2=idx0, period=period, inpct=inpct, lag=lag,
                                     exponential=exponential, span=longwindow, cols=None)
        fdf = pd.DataFrame(data=(acorrshort - acorrlong))
        # fdf = acorrshort - acorrlong
        # print('accorshor:::{}'.format(acorrshort))
        # print('acorlong:::{}'.format(acorrlong))
    elif operation == 'cat':
        quantilize = parameters.get('quantilize', False)
        levels = parameters.get('levels', [-1000, 0, 1000])
        dstart = parameters.get('dstart', None)
        catcols = parameters.get('catcols', None)
        dend = parameters.get('dend', None)
        type_quant = parameters.get('type_quantilize', '')

    elif operation == 'modifdur':
        maturity = parameters.get('maturity', 1)

    elif operation == 'cumret':
        timeweight = parameters.get('timeweight', False)
        fdf = dfunc.apply_cumulative_return(df=idx0, timeweight=timeweight)

    elif operation == 'time':
        pass

    if 'mult' in parameters:
        mult = parameters['mult']
        if mult != 1:
            return fdf * mult

    if 'lag' in parameters:
        lag = parameters['lag']
        if lag != 0:
            return dfunc.apply_lag(fdf, lag=lag, freq=freq)
    if 'add' in parameters:
        add = parameters['add']
        if add != 0:
            return fdf + parameters['add']

    if 'power' in parameters:
        power = parameters['power']

    if 'levels' in parameters and operation != 'cat' :
        if levels > 0:
            quantilize = parameters['quantilize']
            dstart = parameters['dstart']
            catcols = parameters['catcols']
            dend = parameters['dend']

    return fdf
