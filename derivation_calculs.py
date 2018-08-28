# -*- coding: utf-8 -*-
import pandas as pd
import derivation_functions as dfunc
import numpy as np
import datetime

glbnano = 86400000000000.0


def read_df(x):
    return x.read_var(x.get_param('path'))


def get_var_name(x):
    return x.get_param('var_name')


def apply_operation(var_list, freq, operation, parameters, histodata):
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
                The derivation to apply to the list of variables
 
    parameters : {Dict type}
                 The parameters of the derivation

    Return
    ------
    output_df : {Dataframe type}
            The output dataframe,
            Is empty if operation not found
    """
    output_df = pd.DataFrame()
    # load dataframes
    dfs = map(lambda x: read_df(x), var_list)
    # var_names = map(lambda x: get_var_name(x), var_list)
    idx0 = dfs[0]
    # var_name0 = var_names[0]
    idx1 = dfs[1] if len(list(dfs)) == 2 else None
    # var_name1 = var_names[1] if len(list(var_names)) == 2 else None
    # col1 = parameters.get('col1', 0)
    # if col1 == 1:
    # idx0 = idx1
    # idx1 = idx0
    colout = parameters.get('col_out', None)
    # print('colout')
    # print(type(colout))
    # print(colout)
    # if operation == 'timeshift':
    #     shift = parameters.get('shift', 0)
    #     if shift != 0 or freq is not None:
    #         output_df = dfunc.apply_timeshift(idx0, shift=shift, freq=freq, histodata=histodata)
    #         # if idx1 is not None:
    #         #     idx1 = dfunc.apply_timeshift(idx1, shift=shift, freq=freq, histodata=df_derived)

    # if operation == 'timeshift':
    #     shift = parameters.get('shift', 0)
    #     output_df = idx0
    fill_rules = map(lambda x: x.get_param('fill_rule'), var_list)

    if len(fill_rules) > 1:
        fill_rule = fill_rules[1]
    else:
        fill_rule = 'ffill'
    if 'shift' in parameters:

        shift = parameters['shift']
        if shift != 0 or freq is not None:
            idx0 = dfunc.apply_timeshift(idx0, shift=shift, freq=freq)
            if idx1 is not None:
                idx1 = dfunc.apply_timeshift(idx1, shift=shift, freq=freq)
    if operation == 'timeshift':
        shift = parameters.get('shift', 0)
        output_df = idx0

    elif operation == 'fillmissing':
        idxmain = parameters.get('main', 0)
        idxsubst = parameters.get('subst', 1)
        output_df = dfunc.fill_missing_values(idxmain=idx0, idxsubst=idx1)

    elif operation == 'combi':
        coeff1 = parameters.get('coeff1', 1)
        coeff2 = parameters.get('coeff2', 0)
        islinear = parameters.get('lin', True)
        transfo = parameters.get('transfo', None)
        idx1 = parameters.get('col1', 0)
        idx2 = parameters.get('col2', 1)

        output_df = dfunc.apply_combi(df1=dfs[0], df2=dfs[1], idx1=idx1, idx2=idx2, coeff1=coeff1, coeff2=coeff2,
                                      islinear=islinear, transfo=transfo, histodata=histodata)

    elif operation == 'pctdelta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        output_df = dfunc.take_diff(idx0, period=period, ownfreq=ownfreq, inpct=True)

    elif operation == 'delta':
        period = parameters.get('period', 1)
        ownfreq = parameters.get('freq', 'B')
        output_df = dfunc.take_diff(df=idx0, period=period, inplace=False,
                                    inpct=False, ownfreq=freq)

    elif operation == 'rollingreturn':
        period = parameters.get('period', 1)
        rollfreq = parameters.get('rollfreq', 'B')
        iweek = parameters.get('iweek', 1)
        iday = parameters.get('iday', 1)
        iroll_interval = parameters.get('iroll_interval', 0)
        output_df = dfunc.apply_rolling(maincol=idx0, substcol=idx1, rollfreq=rollfreq,
                                        iweek=iweek, iday=iday, effectiveroll_lag=iroll_interval,
                                        inpct=True)

    elif operation == 'ewma':
        emadecay = parameters.get('emadecay', 2.0 / (20 + 1))
        wres = parameters.get('wres', True)
        # wz = parameters.get('wZ', True)
        output_df = dfunc.apply_ewma(df=idx0, emadecay=emadecay, wres=wres, inplace=True,
                                     normalize=True, stdev_min=1e-5, histoemadata=histodata)

    elif operation == 'futuresroll':
        rolldict = {'freq': parameters.get('freq', 'B'),
                    'week': parameters.get('week', 0),
                    'weekday': parameters.get('weekday', 0),
                    'day': parameters.get('day', -1),
                    'bday_offset': parameters.get('bday_offset', 0),
                    'bmonth_offset': parameters.get('bmonth_offset', 0)
                    }
        output_df = dfunc.apply_futures_roll(col_c1=idx0, col_c2=idx1, roll_dict=rolldict)

    elif operation == 'vol':
        period = parameters.get('period', 1)
        window = parameters.get('window', 20)
        inpct = parameters.get('inpct', True)
        annualize = parameters.get('annualize', True)
        fillinit = parameters.get('fillinit', True)
        output_df = dfunc.apply_vol(df=idx0, period=period, window=window, inpct=inpct,
                                    annualize=annualize, fillinit=fillinit)

    elif operation == 'ohlcvol':
        period = parameters['period']
        window = parameters['window']
        inpct = parameters['inpct']
        annualize = parameters['annualize']
        fillinit = parameters['fillinit']
        algo = parameters['algo']
        columns = parameters['columns']
        output_df = dfunc.apply_ohlc_vol(df=idx0, OHLCcols=columns,
                                         window=window, inpct=inpct,
                                         annualize=annualize,
                                         fillinit=fillinit,
                                         algo=algo)
    elif operation == 'corr':
        period = parameters.get('period', 1)
        window = parameters.get('window', 20)
        lag = parameters.get('lag', 0)
        inpct = parameters.get('inpct', True)
        exponential = parameters.get('exponential', True)
        output_df = dfunc.apply_corr(df1=idx0, df2=idx1, period=period, inpct=inpct,
                                     lag=lag,
                                     exponential=exponential, span=window,
                                     fill_rule=fill_rule)
    elif operation == 'delta_acorr':
        period = parameters.get('period', 1)
        shortwindow = parameters.get('shortwindow', 20)
        longwindow = parameters.get('longwindow', 100)
        lag = parameters.get('lag', 0)
        inpct = parameters.get('inpct', True)
        exponential = parameters.get('exponential', True)
        acorrshort = dfunc.apply_corr(df1=idx0, df2=idx0, period=period, inpct=inpct, lag=lag,
                                      exponential=exponential, span=shortwindow)
        acorrlong = dfunc.apply_corr(df1=idx0, df2=idx0, period=period, inpct=inpct, lag=lag,
                                     exponential=exponential, span=longwindow)
        output_df = pd.DataFrame(data=(acorrshort - acorrlong))

    elif operation == 'cat':
        quantilize = parameters.get('quantilize', False)
        levels = parameters.get('levels', [-1000, 0, 1000])
        dstart = parameters.get('dstart', None)
        catcols = parameters.get('catcols', None)
        dend = parameters.get('dend', None)
        type_quant = parameters.get('type_quantilize', '')
        if type(levels) == list or levels > 0:
            if type_quant == 'auto':
                output_df = dfunc.auto_categorize(idx0, mod=10, date_end=dend, min_r=0.02)
            else:
                output_df = dfunc.categorize(idx0, quantilize=quantilize, levels=levels,
                                             cols=catcols, dstart=dstart, dend=dend, inplace=False)
    elif operation == 'modifdur':
        maturity = parameters.get('maturity', 1)
        output_df = dfunc.calc_modified_duration(idx0, n=maturity)
    elif operation == 'cumret':
        timeweight = parameters.get('timeweight', False)
        output_df = dfunc.apply_cumulative_return(df=idx0, timeweight=timeweight)

    elif operation == 'time':
        output_df = dfunc.time_columns(idx0)
    #
    # if type(colout) == 'int':
    #     # else:
    #     col = output_df.columns[colout]
    #     output_df = output_df[col]
    # elif type(colout) == 'str':
    #     #     col = colout[0]
    #     #     output_df = output_df[col]
    #     # else:
    #     output_df = output_df[colout]
    # elif type(colout) == 'list':
    #     output_df = output_df[output_df.columns[colout]]
        # output_df = output_df.loc[:, output_df.columns.isin(colout)]

        # col = colout[0]
        # print('col {}'.format(col))
        # output_df = output_df[col]
    # else:
    #     output_df = output_df
    # print(output_df)
    if 'mult' in parameters:
        mult = parameters['mult']
        if mult != 1:
            if histodata is not None:
                if output_df.index[-1] > histodata.index[-1]:
                    dend = output_df.index[-1]
                    dstart = histodata.index[-1]
                    df_calc = dfunc.take_interval(output_df, dstart=dstart, dend=dend, inplace=True)
                    df_calc = df_calc * mult
                    output_df = histodata.append(df_calc)
                    output_df = output_df[~output_df.index.duplicated(take_last=False)]

                else:
                    output_df = histodata
            else:
                output_df = output_df * mult

    if 'lag' in parameters:
        lag = parameters['lag']
        if lag != 0:
            output_df = dfunc.apply_lag(output_df, lag=lag, freq=freq)
    if 'add' in parameters:
        add_val = parameters['add']
        if add_val != 0:
            if histodata is not None:
                if output_df.index[-1] > histodata.index[-1]:
                    dend = output_df.index[-1]
                    dstart = histodata.index[-1]
                    df_calc = dfunc.take_interval(output_df, dstart=dstart, dend=dend, inplace=True)
                    df_calc = df_calc + add_val
                    output_df = histodata.append(df_calc)
                    output_df = output_df[~output_df.index.duplicated(take_last=False)]

                else:
                    output_df = histodata
            else:
                output_df = output_df + add_val
    if 'power' in parameters:
        power = parameters['power']
        if power != 1:
            if histodata is not None:
                if output_df.index[-1] > histodata.index[-1]:
                    dend = output_df.index[-1]
                    dstart = histodata.index[-1]
                    df_calc = dfunc.take_interval(output_df, dstart=dstart, dend=dend, inplace=True)
                    df_calc = df_calc ** power
                    output_df = histodata.append(df_calc)
                    output_df = output_df[~output_df.index.duplicated(take_last=False)]

                else:
                    output_df = histodata
            else:
                output_df = output_df ** power

    if 'levels' in parameters and operation != 'cat':
        if levels > 0:
            quantilize = parameters['quantilize']
            dstart = parameters['dstart']
            catcols = parameters['catcols']
            dend = parameters['dend']
            output_df = dfunc.categorize(quantilize=quantilize, levels=levels,
                                         cols=catcols, dstart=dstart, dend=dend)
    if 'apply_filter' in parameters:
        if parameters['apply_filter'] is True:
            period = parameters.get('period', 0)
            min_value = parameters.get('min_value', np.NINF)
            max_value = parameters.get('max_value', np.inf)
            diff_order = parameters.get('diff_order', 1)
            inpct = parameters.get('inpct', True)
            output_df = dfunc.apply_filter(idx0, period=period, min_value=min_value,
                                           max_value=max_value, diff_order=diff_order,
                                           inpct=inpct, inplace=True, cols=None)

    output_df = dfunc.exclude_interval(output_df, datetime.datetime(year=2001, month=9, day=11), datetime.datetime(year=2001, month=9, day=18))
    return output_df
