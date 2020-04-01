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
df_patterns = miner.find({'name'     : 'Pattern 1',
     'pattern'  : '-->',
     'P_columns': ['TV-life'],
     'P_values' : [0],
     'Q_values' : [8800,200],
     'Q_columns': ['TV-nonlife', 'Own funds'],
     'parameters' : {"min_confidence" : 0, "min_support" : 1,
                     'Q_operators':['>', '>'],
                     'Q_logics'  : ['|']}} ) 
df_patterns
print(df_patterns.iloc[:,:12].to_string())
print(df_patterns.loc[0,'pandas ex'])
# miner = data_patterns.PatternMiner(df)
# # df_patterns = miner.find({'name'      : 'equal values',
# #                           'pattern'   : '=',
# #                           'parameters': {"min_confidence": 0.5,
# #                                          "min_support"   : 2}})
# # df_patterns = miner.find({'pattern'   : ['<','>'],
# #                           'values'     : [8000,0],
# #                           'P_columns'  : ['Assets'],
# #                           'Q_columns'  : ['TV-life'],
# #                           'parameters': {"min_confidence": 0.4,
# #                                          "min_support"   : 1,
# #                                         'both_ways' : True}})
#
# # df = data_patterns.make_new_columns(df, columns = [['Assets', 'TV-life'],["Assets",'Own funds','Excess']],
# #                                                 operation = [['+'], ['+','*']],
# #                                                 new_names = ['test1','test2'])
# # #print(df)
# # miner = data_patterns.PatternMiner(df)
# # print(df_patterns.iloc[:,:12].to_string())
# # print(df_patterns.loc[0,'pandas ex'])
# # print(df[((df["Assets"]<8000) & ~(df["TV-life"]>0)) | (~(df["Assets"]<8000) & ((df["TV-life"]>0)))])
# print(df[(df["Assets"]<9000) & ~((df["TV-life"]<df["TV-nonlife"]))])
# df_patterns = miner.find({'pattern'   : [['>','<'],['>','<']],
#                           'values'     : [[0,8000], [ 210, 1000]],
#                           'P_columns'  : ['TV-life', 'Assets'],
#                           'parameters': {"min_confidence": 0.0,
#                                          "min_support"   : 1,
#                                           'both_ways' : False,
#                                            'P_logics' : ['&'],
#                                            'Q_logics' : ['^']}})
# print(df_patterns.iloc[:,:12].to_string())
# print(df_patterns.loc[3,'pandas ex'])
# print(df[(df["TV-life"]>0)&(df["Assets"]<8000) & ~((df["Own funds"]>210)^(df["Excess"]<1000))])
