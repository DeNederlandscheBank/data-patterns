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
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
df.set_index('Name', inplace = True)

# p2 = {'name' : 'Pattern 1',
#      'pattern'  : '-->',
#      'P_columns': ['TV-life', 'Assets'],
#      'P_values' : [100,0],
#      'Q_values' : [0,0],
#      'Q_columns': ['TV-nonlife', 'Own funds'],
#      'parameters' : {"min_confidence" : 0, "min_support" : 1, 'Q_operators': ['>', '>'],
#      'P_operators':['<','>']}}
#
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2 )
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
pattern = {'name'     : 'Pattern 1',
           'P_columns': ['Type'],
           'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
           'encode'   : {'Assets':      'reported',
                         'TV-life':     'reported',
                         'TV-nonlife':  'reported',
                         'Own funds':   'reported'}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(pattern)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])

parameters = {'min_confidence': 0.5,'min_support'   : 2}
p2 = {'name'      : 'type pattern',
      'expression' : 'IF ({".*TV-life.*"} = 0) THEN ({".*TV-nonlife.*"} = 8800) AND IF ~({".*TV-life.*"} = 0) THEN ~({".*TV-nonlife.*"} = 8800)',
      'parameters' : parameters }
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p2)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])

parameters = {'min_confidence': 0.5,'min_support'   : 2}
p2 = {'name'      : 'type pattern',
      'expression' : 'IF ({".*Ty.*"} != @) THEN ({".*.*"} = @)'}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p2)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])
