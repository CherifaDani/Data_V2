# -*- coding: utf-8 -*-
import pandas as pd

"""
    This script retrieves all variables with their types
    and saves it to a csv file

"""

from_xls = '/home/cluster/drop_ch/Dropbox/Project Data v2.0/Docs/18 07 Script_Derivation_1V_bis.xlsx'
to_csv =  '/home/cluster/drop_ch/Dropbox/Project Data v2.0/Docs/var_state.csv'


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
