# -*- coding: utf-8 -*-
import pandas as pd


def retrieve_vars():
    """
        This script retrieves all variables with their types
        and saves it to a csv file

    """

    xls = pd.ExcelFile(from_xls, on_demand=True)
    sheets = xls.sheet_names
    print(sheets)
    dfx = pd.DataFrame()
    e = []
    for x in sheets:
        x = x.encode('utf-8')
        if not x.startswith('_'):
            e.append(x)
    # e = ['primary'] # TEST PURPOSE!
    for sheet in e:
        dfs = xls.parse(sheet)
        var_name = dfs['SQL_Name'].values
        cols = ['var_name', 'var_type']
        list1 = []  # list of variable names
        list2 = []  # type of the variables
        for x in var_name:
            list1.append(x)
            list2.append(sheet)
        # Drop duplicates
        list1 = list(set(list1))
        df1 = pd.DataFrame(data=list1, columns=['var_name'])
        df2 = pd.DataFrame(data=list2, columns=['var_type'])
        df = pd.concat([df1, df2], axis=1)
        dfx = pd.concat([dfx, df])
    dfx = dfx.dropna(how='any')

    print(dfx.shape)
    dfx.to_csv(to_csv, index=False)


if __name__ == '__main__':
    from_xls = '/media/D/Advestis Dropbox/Missions_Lab/18 03 1V/2 Data/T3/2 Derived/GenMat/19 06 Derivation 1V.xlsx'
    to_csv = '/media/D/Advestis Dropbox/Missions_Lab/18 03 1V/2 Data/T3/2 Derived/GenMat/var_state.csv'
    retrieve_vars()