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
                    ['Insurer  1', 'life & insurer',     90200,     8800,      0,             200,         200],
                    ['Insurer  1', 'life & insurer',     91000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurerd', 90030,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurer', 90200,     0,         8800,          200,         199.99]])
#
# df['LA'] = 0
#
# df.set_index(['Excess', 'Assets'], inplace=True)
print(df)
p1 = {'name'      : 'equal values',
                              'expression'   : 'IF {"Type"} = [.*insurer@] THEN {"TV-nonlife"} = [@]',
                              'parameters': {"min_confidence": 'highest',
                                             "min_support"   : 1}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p1)

print(miner.df_patterns.to_string())
df_ana = miner.analyze()
print(miner.df_results)

# miner = data_patterns.PatternMiner(df)
#
# p2 = {'name'     : 'Condition',
#      'pattern'  : '-->',
#      'P_columns': ['Name'],
#      'Q_columns': ['Type'],
#      'parameters' : {"min_confidence" : 'highest', "min_support" : 1}}
# df_patterns = miner.find(p2)
# df_ana = miner.analyze()
#
# df_correct = miner.correct_data()
# print(df_patterns.to_string())
# print(df_correct[0])


parameters = {'solvency': True, 'decimal' : 2}

expression = '{"S.28.01.01.03,R0200,C0040"} = MAX(0,(MAX(0, {"S.28.01.01.04,R0210,C0050"} * 0.037) - MAX(0,{"S.28.01.01.04,R0220,C0050"} * 0.052) + MAX(0, {"S.28.01.01.04,R0230,C0050"} * 0.007) + MAX(0, {"S.28.01.01.04,R0240,C0050"} *  0.021) + MAX(0, {"S.28.01.01.04,R0250,C0060"} * 0.0007)))'
pandas_expressions = data_patterns.to_pandas_expressions(expression, {}, parameters, None)
# n_ex = eval(pandas_expressions[0], {}, {'df': df, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum})
print(pandas_expressions[0])
expression = '{"S.28.01.01.03,R0200,C0040"} = 44'
pandas_expressions = data_patterns.to_pandas_expressions(expression, {}, parameters, None)
# n_ex = eval(pandas_expressions[0], {}, {'df': df, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum})
print(pandas_expressions[0])
expression = '{"S.28.01.01.03,R0200,C0040"} = "NET"'
pandas_expressions = data_patterns.to_pandas_expressions(expression, {}, parameters, None)
# n_ex = eval(pandas_expressions[0], {}, {'df': df, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum})
print(pandas_expressions[0])

#
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
