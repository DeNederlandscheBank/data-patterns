import data_patterns
import numpy as np
import pandas as pd
import logging
import os

# logging.basicConfig()
# logging.getLogger().setLevel(logging.WARNING)

df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
          data   = [['Insurer  1', 'life & insurerd',     1000,     800,       0,             200,         200],
                    ['Insurer  2', 'non-life & insurer', 4000,     0,         3200,          800,         800],
                    ['Insurer  2', 'non-life & insurer', 800,      0,         700,           100,         100],
                    ['Insurer  1', 'life & insurer',     2500,     1800,      0,             700,         700],
                    ['Insurer  2', 'non-life & insurer', 2100,     0,         2200,          200,         200],
                    ['Insurer  1', 'life & insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'non-life & insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'life & insurer',     91000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurerd', 90030,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurer', 90200,     0,         8800,          200,         199.99]])

df['LA'] = 0

df.set_index(['Name', 'Assets'], inplace=True)
print(df)
p1 =     {'name'      : 'sum pattern',
               'pattern'   : '-->',
               'P_columns' :['Type'],
               'Q_columns' :['TV-life'],
               'parameters': {"min_confidence": 'highest',
                                      "min_support"   : 2 }}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p1)

print(miner.df_patterns.to_string())
df_ana = miner.analyze()
print(miner.df_results)
#
# p2 = {'name'      : 'Pattern 1',
#     'expression' : 'IF ({.*Ty.*} = "@") THEN ({.*.*} = "@")', 'parameters': {"min_confidence": 0.5,
#                            "min_support"   : 2 }}
#
# df_patterns = miner.find(p2)
#
# print(miner.df_patterns.to_string())
# df_ana = miner.analyze()
# print(miner.df_results)
#
#
#
#
#
#
# pattern ={'name'      : 'sum pattern',
#                           'expression'   : 'IF {"Assets"} <> 0|{"TV-life"}<>0 THEN {"Excess"}<>0',
#                           'parameters': {"min_confidence": 0.2,
#                                          "min_support"   : 1}}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
# df_ana = miner.analyze()
# print(miner.df_results)
# pattern ={'name'      : 'sum pattern',
#                           'expression'   : '{.*} = {.*}',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1, 'decimal':8}}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
# df_ana = miner.analyze()
# print(miner.df_results)
# pattern ={'name'      : 'sum pattern',
#                           'expression'   : '{.*}=0',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1}}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
# df_ana = miner.analyze()
# print(miner.df_results)
# data_patterns.get_value('({"(Technical provisions, Technical provisions - life (excluding non-life and index-linked and unit-linked), Initial absolute values, B, 9.0)"} = {"S.02.01.01.01,r0600,c0010 (S.02.01.01.01,Liabilities|Technical provisions - life (excluding index-linked and unit-linked) , Solvency II value)"})', 2,1)
