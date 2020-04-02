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
     'P_columns': ['TV-life', 'Assets'],
     'P_values' : [100, 'Excess'],
     'Q_values' : [0,0],
     'Q_columns': ['TV-nonlife', 'Own funds'],
     'parameters' : {"min_confidence" : 0, "min_support" : 1, 'Q_operators': ['>', '>'],
     'P_operators':['<','>'], 'Q_logics':['&'], 'both_ways':False}} )

print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas ex'])

miner = data_patterns.PatternMiner(df)
df_patterns = miner.find({'name'     : 'Pattern 1',
     'pattern'  : '-->',
     'P_columns': ['Assets'],
     'P_values' : [0],
     'Q_columns': ['Type', 'TV-life', 'TV-nonlife', 'Own funds', 'Excess'],
     'parameters' : {"min_confidence" : 0, "min_support" : 1,
     'P_operators':['>']}} )

print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])

miner = data_patterns.PatternMiner(df)

df_patterns = miner.find({'name'      : 'equal values',
                          'pattern'   : '>',
                          'value' : 0,
                          'parameters': {"min_confidence": 0.5,
                                         "min_support"   : 2,
                                         "decimal"       : 0}})
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])

miner = data_patterns.PatternMiner(df)
df_patterns = miner.find({'name'      : 'sum pattern',
                          'pattern'   : 'sum',
                          'parameters': {"min_confidence": 0.5,
                                         "min_support"   : 1}})

print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])
