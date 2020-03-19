import data_patterns.data_patterns
import numpy as np
import pandas as pd
df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     0,         8800,          200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     0,         8800,          200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
df.set_index('Name', inplace = True)

miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find({'name'      : 'sum pattern',
#                           'pattern'   : 'sum',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "sum_elements": 3}})
df_patterns = miner.find({'pattern'   : ['=','>'],
                          'values'     : ['life insurer',0],
                          'P_columns'  : ['Type'],
                          'Q_columns'  : ['TV-life'],
                          'both_ways' : True,
                          'parameters': {"min_confidence": 0.5,
                                         "min_support"   : 1}})
print(df_patterns.to_string())
# df_patterns = miner.find({'name'      : 'higher than',
#                           'pattern'   : '>',
#                           'value'     : 1000,
#
#                           'parameters': {"min_confidence": 0.8,
#                                          "min_support"   : 1}})
#
# print(df_patterns.iloc[:,:12].to_string())

# #
# data_path = r"C:\Users\jan_h_000\Documents\DNB\Python"
# xls = pd.ExcelFile(data_path + r"\Data individual insurers (year).xlsx")\
#
# def get_sheet(num):
#     # read entire Excel sheet
#     df = xls.parse(num)
#     # columns names to lower case
#     df.columns = map(str.lower, df.columns)
#     # set index to name and period
#     df.set_index(['relatienaam', 'periode'], inplace = True)
#     # data cleaning (the excel sheet contains some
#                     # additional data that we don't need)
#     drop_list = [i for i in df.columns
#                      if 'unnamed' in i or 'selectielijst' in i]
#     df.drop(drop_list, axis = 1, inplace = True)
#     # pivot data frame
#     if "row_name" in df.columns:
#         df.drop("row_name", axis = 1, inplace = True)
#         df = df.pivot(columns = 'row_header')
#     if df.columns.nlevels > 1:
#         df.columns = [str(df.columns[i]) for i in
#                           range(len(df.columns))]
#     return df
# df1 = get_sheet(14)
# df2 = get_sheet(18)
# df2.columns = [str(df2.columns[i]) for i in range(len(df2.columns))]
# miner = data_patterns.PatternMiner(df1)
#
# df_patterns = miner.find({'name'      : 'sum pattern',
#                           'pattern'   : 'sum',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "sum_elements": 3}})
# print(df_patterns[['P columns','Q columns', 'support','exceptions']])
