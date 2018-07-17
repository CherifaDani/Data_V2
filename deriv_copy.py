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


def apply_corr(df, period=1, exponential=True, inpct=True, lag=0, span=20):
    """

    :param df:
    :param period:
    :param span:
    :param exponential:
    :param inpct:
    :param lag:
    :return:
    '''Renvoie la série des corrélations entre deux colonnes d'un Dataset
           period: si 0, corrélation des valeurs, si > 0, corrélation des
           variations sur period
           lag: retard sur la seconde colonne
           cols: spécifications de 1 ou 2 colonnes
    rolling_corr(data1, data2, window=span)
"""

    # #  si period == 0 c'est l'autocorrélation des valeurs
    # #  et non des variations qui est calculée
    startval = period + lag * period
    if period == 0:
        data1 = df
        data2 = df.shift(periods=lag)
    else:
        if inpct:
            data1 = df.pct_change(period)[startval:]
            data2 = df.pct_change(period).shift(periods=lag * period)[startval:]
        else:
            data1 = df.diff(period)[startval:]
            data2 = df.diff(period).shift(periods=lag * period)[startval:]

    if exponential:
        corrdata = pd.ewmcorr(data1[startval:], data2[startval:], span=span)
    else:
        corrdata = pd.rolling_corr(data1, data2, window=span)

    corrdata = corrdata.dropna()
    return corrdata


def apply_pctdelta(df, period, freq, inpct):
    if inpct:
        deltadata = df.pct_change(period)
    else:
        deltadata = df.diff(period)

    idx_all = pd.bdate_range(start=(deltadata.index[0]).date(),
                             end=(deltadata.index[-1]).date(),
                             freq=freq)

    # Reindex using datetime index, to drop hours and minutes
    deltadata.index = pd.DatetimeIndex(deltadata.index).normalize()
    if (freq == 'B' or freq == 'D'):
        deltadata = deltadata.reindex(index=idx_all, method=None)

    else:
        deltadata = deltadata.reindex(index=idx_all, method='pad')

    return deltadata


def apply_rolling(df1, df2, rollfreq, iweek, iday, iroll_interval, freq):
    """
        Renvoie la série des variations d'une colonne pour un décalage donné.
        Dans le calcul de V(t) / V(t - p), V est la série principale self [maincol].
        Par exception, aux dates spécifiées par la règle rolldate, on calcule V(t) / Vsubst(t-p),
        où Vsubst représente la série self [substcol]
    """
    # élargir le calendrier pour inclure les dates de rolls de façon certaine
    idx_all = pd.bdate_range(start=(df1.index[0]).date(),
                             end=(df1.index[-1]).date(),
                             freq=freq)
    df1.index = pd.DatetimeIndex(df1.index).normalize()
    data = df1.reindex(index=idx_all, method=None)
    rolldates = pd.bdate_range(data.index[0], data.index[-1], freq=rollfreq)
    rolldates = rolldates + pd.datetools.WeekOfMonth(week=iweek, weekday=iday)


def apply_ewma(df, emadecay, wres, wz):
    """
    Renvoie la série des ema d un ensemble de colonnes pour une pseudo durée(span) donnée
     self: contient la totalité des données primaires dont on veut calculer la moyenne
    emadecay: coefficient d'atténuation de la moyenne(proche de 1). Prioritaire si fourni.
    span: 2/emadecay - 1
    cols: groupe de colonnes dont on calcule l'ewma.
    wres: si True, on calcule également le résidu
    normalize: si True, on calcule aussi le Z-Score(résidu / ewmastd(même span))
    histoemadata: série facultative contenant les valeurs historiques de l'ewma sur des dates
       normalement antérieures aux données primaires.
    overridedepth: nombre de jours passés(à partir de la donnée la plus récente) à recalculer
    """
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

    if not wres:
        return df_calc
    elif wz is True:
        return zcols
    else:
        return rescols


def apply_fill(df1, df2, dfsubst):
    return None


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
    # les dates de fin de périodes(mois ou trimestre ouvré)
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


def apply_futures_roll(df1, df2, rolldict):
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
    df = pd.concat([df1, df2], axis=1)
    df.dropna(inplace=True)
    h_c1, m_c1, s_c1 = df.index[0].hour, df.index[0].minute, df.index[0].second
    dstart = df.index[0]
    dend = df.index[-1]
    bdates = pd.bdate_range(dstart, dend, freq='B')
    bdates = map(lambda d: d.replace(hour=h_c1, minute=m_c1, second=s_c1), bdates)
    # les dates de roll
    df_rolldates = apply_roll_shift(bdates, rolldict)
    # les rendements quotidiens
    df1 = df[df.columns[0]]
    df2 = df[df.columns[1]]

    df_ret1 = df1.dropna().pct_change(periods=1)
    df_ret1.columns = ['RETURN_1D']
    # rendements modifiés après prise en compte du roll
    # df_ret1['RETURN_1D_AFTER_ROLL'] = df_ret1['RETURN_1D']
    # le jour suivant le roll dans le calendrier du contrat
    df_ret1['NEXT_BDATE'] = np.nan
    df_ret1['NEXT_BDATE'][:-1] = df_ret1.index[1:]
    # les cours des 2 premiers contrats
    df_ret1['C1'] = df1.dropna()
    df_ret1['C2'] = df2.dropna()
    # # le ratio c1 / c2 prolongé à tous les jours cotés
    # df_ret1['RATIO12'] = (df_ret1['C1'] / df_ret1['C2']).fillna(method='ffill')
    # next_bd_after_roll = df_ret1.loc[df_rolldates['LAST_TRADING_DATE'],
    #                                  'NEXT_BDATE'].fillna(method='ffill').fillna(method='bfill')
    # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = 1.0 + df_ret1.loc[
    #     next_bd_after_roll, 'RETURN_1D_AFTER_ROLL']
    # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] *= df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] * \
    #                                                            df_ret1.loc[
    #                                                                df_rolldates['LAST_TRADING_DATE'], 'RATIO12'].values
    # df_ret1.loc[next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] = df_ret1.loc[
    #                                                               next_bd_after_roll, 'RETURN_1D_AFTER_ROLL'] - 1.0
    #
    # df_roll = pd.DataFrame(index=df_ret1.index, columns=['C1', 'C1_ROLL', 'RETURN_1D_AFTER_ROLL', 'RETURN_1D', 'C2'])
    # df_roll['C1_ROLL'] = 1.0
    #
    # roll_ret = np.log(1 + df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL'])
    # roll_ret = pd.expanding_sum(roll_ret)
    # roll_nav = np.exp(roll_ret)
    # df_roll['C1_ROLL'] = roll_nav * df.iloc[0, 0]
    # df_roll['C1'] = df.take_columns(col_c1)
    # df_roll['RETURN_1D_AFTER_ROLL'] = df_ret1.loc[:, 'RETURN_1D_AFTER_ROLL']
    # df_roll['RETURN_1D'] = df_ret1.loc[:, 'RETURN_1D']
    # df_roll['C2'] = df_ret1.loc[:, 'C2']
    return df_ret1


def estimate_nat_freq(df):
    '''Estime la fréquence naturelle d'une série: la fréquence des changements de valeur '''
    df.dropna()
    df.sort_index(inplace=True)
    fl = float((df.index.asi8[-1] - df.index.asi8[0]) / glbnano)

    # série des différences
    ddf = df.diff(1)
    # série des différences non nulles
    ddf = df[ddf != 0]
    # rajouter une colonne pour les différences de dates
    ddf['deltat'] = 0
    ddf.deltat[1:] = (ddf.index.asi8[1:] - ddf.index.asi8[: -1]) / glbnano
    # trier les intervalles entre changements de dates
    lastdelta = ddf.ix[-1]
    ddf.sort(columns='deltat', inplace=True)
    l = len(ddf)
    deltat = ddf.deltat[1:]
    fdict = {}

    if l > 1:
        fdict['last'] = lastdelta
        fdict['min'] = mind = deltat.min()
        fdict['datemin'] = deltat.idxmin()
        fdict['pct5'] = mind
        fdict['pct10'] = mind
        fdict['pct25'] = mind
        fdict['median'] = deltat.ix[int(0.5 * l) - 1]
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
        if l > 4:
            fdict['pct25'] = deltat.ix[int(0.25 * l) - 1]
            fdict['pct75'] = deltat.ix[int(0.75 * l) - 1]
            if l > 10:
                fdict['pct10'] = deltat.ix[int(0.1 * l) - 1]
                fdict['pct90'] = deltat.ix[int(0.9 * l) - 1]
                if l > 20:
                    fdict['pct5'] = deltat.ix[int(0.05 * l) - 1]
                    fdict['pct95'] = deltat.ix[int(0.95 * l) - 1]
    return fdict


def apply_vol(df, period, window, inpct, annualize, fillinit, freq):
    '''Renvoie la série des volatilités de rendements '''
    if period != 0:
        diffdata = apply_pctdelta(df, period, freq, inpct)
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
            nfreqdict = estimate_nat_freq(col)
            nfreq = max(1, nfreqdict['min'])
            annfactor = math.sqrt(260 / nfreq)
        else:
            annfactor = 1
        newdf = voldata[voldata.columns[icol]] * annfactor

    return newdf


def apply_ohlcvol(df):
    pass


def apply_cumulative_return(df, timeweight=False):
    '''Renvoie le cumul composé des rendements'''
    '''AMBIGU quand inplace=True, cols <> None'''
    if timeweight is True:
        deltatime = pd.Series(df.index.asi8)
        deltatime = deltatime.diff(1) / glbnano
        deltatime.fillna(value=0.0, inplace=True)
        deltatime = deltatime / 365.25
        deltatime = deltatime.reshape(len(df.index), 1)
        df = df * deltatime
    navdata = np.log(1 + df)
    navdata = pd.expanding_sum(navdata)
    navdata = np.exp(navdata)
    return navdata

