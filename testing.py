import data_patterns
import numpy as np
import pandas as pd
import logging
import os

# logging.basicConfig()
# logging.getLogger().setLevel(logging.WARNING)

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
# df['LA'] = 0
#
# df.set_index(['Excess', 'Assets'], inplace=True)
print(df)
p1 = {'name'     : 'Pattern 1',
         'pattern'  : '-->',
         'P_columns': ['Type'],
         'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
         'encode'   : {'Assets'   : 'reported',
                      'TV-life'   : 'reported',
                      'TV-nonlife': 'reported',
                      'Own funds' : 'reported'}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p1)

print(miner.df_patterns['pattern_def'][1])
df_ana = miner.analyze()
print(miner.df_results)

miner = data_patterns.PatternMiner(df)

p2 = {'name'     : 'Condition',
     'pattern'  : '-->',
     'P_columns': ['Name'],
     'Q_columns': ['Type'],
     'parameters' : {"min_confidence" : 'highest', "min_support" : 1}}
df_patterns = miner.find(p2)
df_ana = miner.analyze()

# df_correct = miner.correct_data()
print(df_patterns.to_string())

df_ana = miner.analyze()
print(miner.df_results)


pattern ={'name'      : 'sum pattern',
                          'expression'   : '{.*} = {.*}',
                          'parameters': {"min_confidence": 0.5,
                                         "min_support"   : 1, 'decimal':8}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(pattern)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])
df_ana = miner.analyze()
print(miner.df_results)
pattern ={'name'      : 'sum pattern',
                          'expression'   : '{.*}=0',
                          'parameters': {"min_confidence": 0.5,
                                         "min_support"   : 1}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(pattern)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])
df_ana = miner.analyze()
print(miner.df_results)
