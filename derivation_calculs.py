# -*- coding: utf-8 -*-
import pandas as pd
import derivation_functions as dfunc

glbnano = 86400000000000.0


def read_df(x):
    return x.read_var(x.get_param('path'))


def apply_operation(var_list, list_var_name,  freq, operation, parameters):
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
    idx0 = dfs[0] if len(list(dfs)) == 1 else None
    idx1 = dfs[1] if len(list(dfs)) == 2 else None
    if operation == 'timeshift':
        shift = parameters.get('shift', 0)
        fdf = dfunc.apply_timeshift(idx0, freq, shift)

    elif operation == 'corr':
        pass

    elif operation == 'combi':
        # dfs = map(lambda x: read_df(x), var_list)
        coeff1 = parameters.get('coeff1', 1)
        coeff2 = parameters.get('coeff2', 0)
        islinear = parameters.get('lin', True)
        transfo = parameters.get('transfo', None)
        fdf = dfunc.apply_combi(idx0, idx1, coeff1, coeff2, islinear, transfo)

    elif operation == 'pctdelta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        fdf = dfunc.apply_pctdelta(idx0, period=period, freq=ownfreq, inpct=True)

    elif operation == 'delta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        fdf = dfunc.apply_pctdelta(idx0, period=period, freq=ownfreq, inpct=False)

    elif operation == 'rollingreturn':
        period = parameters.get('period', 1)
        rollfreq = parameters.get('rollfreq', 'B')
        iweek = parameters.get('iweek', 1)
        iday = parameters.get('iday', 1)
        iroll_interval = parameters.get('iroll_interval', 0)
        fdf = dfunc.apply_rolling(idx0, idx1, rollfreq, iweek, iday, iroll_interval)

    elif operation == 'ewma':
        emadecay = parameters.get('emadecay', 2.0 / (20 + 1))
        wres = parameters.get('wres', True)
        wz = parameters.get('wZ', True)
        fdf = dfunc.apply_ewma(idx0, emadecay, wres, wz)

    elif operation == 'fillmissing':
        fdf = dfunc.apply_fill()

    elif operation == 'futuresroll':
        rolldict = {'freq': parameters.get('freq', 'B'),
                    'week': parameters.get('week', 0),
                    'weekday': parameters.get('weekday', 0),
                    'day': parameters.get('day', -1),
                    'bday_offset': parameters.get('bday_offset', 0),
                    'bmonth_offset': parameters.get('bmonth_offset', 0)
                    }
        # fdf = apply_futures_roll(idx0, idx1, rolldict)

    elif operation == 'vol':
        period = parameters.get('period', 1)
        window = parameters.get('window', 20)
        inpct = parameters.get('inpct', True)
        annualize = parameters.get('annualize', True)
        fillinit = parameters.get('fillinit', True)
        fdf = dfunc.apply_vol(idx0, period, window, inpct, annualize, fillinit, freq)

    elif operation == 'ohlcvol':
        period = parameters['period']
        window = parameters['window']
        inpct = parameters['inpct']
        annualize = parameters['annualize']
        fillinit = parameters['fillinit']
        algo = parameters['algo']
        columns = parameters['columns']

    elif operation == 'corr':
            period = parameters['period']
            window = parameters['window']
            inpct = parameters['inpct']
            exponential = parameters['exponential']
            lag2 = parameters['lag2']

    elif operation == 'delta_acorr':
        period = parameters.get('period', 0)
        shortwindow = parameters.get('shortwindow', 20)
        longwindow  = parameters.get('longwindow', 100)
        lag = parameters.get('lag', 1)
        inpct = parameters.get('inpct', True)
        exponential = parameters.get('exponential', True)
        acorrshort = dfunc.apply_corr(idx0, period, inpct, lag, exponential, span=shortwindow)

        acorrlong = dfunc.apply_corr(idx0, period, inpct, lag, exponential, span=longwindow)

        fdf = acorrshort - acorrlong

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
        fdf = dfunc.apply_cumulative_return(idx0, timeweight)

    elif operation == 'time':
        pass

    if 'mult' in parameters:
        mult = parameters['mult']
        if mult != 1:
            return fdf * mult

    if 'lag' in parameters:
        return dfunc.apply_timeshift(fdf, freq, parameters['lag'])
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
