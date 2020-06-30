import data_patterns.data_patterns
import numpy as np
import pandas as pd
df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
          data   = [['Insurer  1', 'life insurer.x',     1000,     800,       0,             200,         200],
                    ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                    ['Insurer  2', 'non-life insurer', 800,      0,         700,           100,         100],
                    ['Insurer  1', 'life insurer',     2500,     1800,      0,             700,         700],
                    ['Insurer  2', 'non-life insurer', 2100,     0,         2200,          200,         200],
                    ['Insurer  1', 'life insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'life insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'life insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life insurer', 9000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life insurerd', 9000,     0,         8800,          200,         199.99]])


df.set_index('Name', inplace = True)
df = df.reset_index()

parameters = {'min_confidence': 0,'min_support'   : 1}
p2 ={'name'      : 'sum pattern',
                          'pattern'   : '-->',
                          'P_columns' :['Name'],
                          'Q_columns' :['Type'],
                          'parameters': {"min_confidence": 0,
                                         "min_support"   : 1 }}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p2)

print(miner.df_data)
print(miner.df_patterns)
print(miner.metapatterns)

df_d = miner.add_columns()
print(df_d)

df_ana = miner.analyze(p2)
print(miner.df_results)

df_data = miner.correct_data()

print(df_data.to_string())
# df_res = miner.incorrect_columns()
#
# print(df_res.to_string())

#
# df.set_index('Name', inplace = True)
# parameters = {'min_confidence': 0.5,'min_support'   : 1}
# p2 ={'name'      : 'sum pattern',
#                           'pattern'   : 'sum',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "nonzero" : True }}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2 )
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
#
#
#
# parameters = {'min_confidence': 0,'min_support'   : 0}
#
# p2 = {'name'      : 'equal values',
#                           'pattern'   : '=',
#                           'value' : 0,
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "decimal" : -5 }}
#
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2 )
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
# #
#
#
#
# pattern ={'name'      : 'sum pattern',
#                           'expression'   : '{.*} + {.*} = {.*}',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "nonzero" : True}}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
# #
# pattern ={'name'      : 'sum pattern',
#                           'expression'   : '{.*} = {.*}',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "decimal" : -5,
#                                          'nonzero': True}}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
#
# parameters = {'min_confidence': 0.3,'min_support'   : 1, 'percentile' : 90}
# p2 = {'name'      : 'Pattern 1',
#     'pattern' : 'percentile',
#     'columns' : [ 'TV-nonlife', 'Own funds'],
#   'parameters':parameters}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
