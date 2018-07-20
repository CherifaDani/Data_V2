# coding: utf-8

import pandas as pd
import numpy as np
import logging
from data_utils import reindex
try:
    from var_logger import setup_logging
except ImportError:
    print ("Don't find the package var_logger")

# Launching the logger module
setup_logging()
logger = logging.getLogger(__name__)
logger.debug('Logger for class ')
# Setting the DEBUG level of the logger
logger.setLevel(logging.DEBUG)


def compare_two_dfs(df1, df2):
    """
    Function used to find a compare between two dataframes

    Parameters
    ----------
    df1 : {Dataframe type}
            The DF to compare

    df2 : {Dataframe type}
            The DF to compare

    Return
    ------
    df : {Dataframe type}
            Empty if df1 equals df2, else:
            All different rows [values/ indexes]
    """
    # print(df1, df2)
    df = pd.DataFrame()
    df1 = df1.dropna()
    df2 = df2.dropna()
    # Reindixing the DFs to have the same index
    dfx, dfy = reindex(df1, df2)
    dfx = np.around(dfx, 6)
    dfy = np.around(dfy, 6)

    if df1.equals(df2):
        # Do nothing
        logger.info('DF1 equals DF2')
    else:
        logger.debug('DF1 is different of DF2')
        differences = (dfx != dfy).stack()
        df_diff = differences[differences]
        df_diff.index.names = ['id', 'col']
        difference_locs = np.where(dfx != dfy)
        df_from = dfx.values[difference_locs]
        df_to = dfy.values[difference_locs]
        df = pd.DataFrame({'DF0': df_from, 'DF1': df_to}, index=df_diff.index)
        logger.debug('Comparison done!, There are: {} differences'.
                     format(df.shape[0]))
    return df

