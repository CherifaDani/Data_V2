# coding: utf-8

import pandas as pd
import numpy as np
import logging
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
    df = pd.DataFrame()
    # Reindixing the DFs to have the same index
    if df1.shape[0] > df2.shape[0]:
        dfx = df2.reindex_like(df1)
        dfy = df1
    else:
        dfx = df1.reindex_like(df2)
        dfy = df2

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

    return df
