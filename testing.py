import data_patterns.data_patterns
import numpy as np
import pandas as pd
col = ['Name', 'Type', 'Assets', 'TV-life', 'TV-nonlife', 'Own funds', 'Diversification','Excess']
insurers = [['Insurer  1', 'life insurer',     1000,  800,    0,  200,   12,  200],
            ['Insurer  2', 'non-life insurer',   40,    0,   32,    8,    9,    8],
            ['Insurer  3', 'non-life insurer',  800,    0,  700,  100,   -1,  100],
            ['Insurer  4', 'life insurer',       25,   18,    0,    7,    8,    7],
            ['Insurer  5', 'non-life insurer', 2100,    0, 2200,  200,   12,  200],
            ['Insurer  6', 'life insurer',      907,  887,    0,   20,    7,   20],
            ['Insurer  7', 'life insurer',     7123,    0, 6800,  323,    5,  323],
            ['Insurer  8', 'life insurer',     6100, 5920,    0,  180,   14,  180],
            ['Insurer  9', 'non-life insurer', 9011,    0, 8800,  211,   19,  211],
            ['Insurer 10', 'non-life insurer', 1034,    0,  901,  133,    1,  134]]
df = pd.DataFrame(columns = col, data = insurers)
df.set_index('Name', inplace = True)
df
parameters = {'min_confidence': 0.2,'min_support'   : 1, 'nonzero':False,'solvency' : True}

p2 = {'name'      : 'Pattern 1',
      'expression' : 'IF {"S.27.01.01.04,R1260,C0220"}<>0 THEN {"S.27.01.01.04,R1260,C0240"}={"S.27.01.01.04,R1260,C0230"}/{"S.27.01.01.04,R1260,C0220"}',
      'parameters' : parameters }

miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p2 )
print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
#
# parameters = {'min_confidence': 0.2,'min_support'   : 1, 'nonzero':False}
# p2 = {'name'      : 'type pattern',
#         'pattern' : '=',
#         'value' : '"@"',
#         # 'columns' : [ 'TV-nonlife', 'Own funds', 'Diversification'],
#       'parameters':parameters}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
#
# parameters = {'min_confidence': 0.3,'min_support'   : 1, 'percentile' : 90}
# p2 = {'name'      : 'type pattern',
#         'pattern' : 'percentile',
#         'columns' : [ 'TV-nonlife', 'Own funds', 'Diversification'],
#       'parameters':parameters}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p2)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
#
# pattern ={'name'      : 'sum pattern',
#                           'pattern'   : 'sum',
#                           'parameters': {"min_confidence": 0.5,
#                                          "min_support"   : 1,
#                                          "nonzero" : True }}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(pattern)
# print(df_patterns.to_string())
# print(df_patterns.loc[0,'pandas co'])
