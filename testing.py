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


p1 = {'name'     : 'Pattern 1',
         'pattern'  : '='}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(p1)

print(df_patterns.to_string())



df = pd.DataFrame(columns = ['Year',       'Name',   'volgnummer' ,         '1', '2', '3' , '4', '5'],
                  data   = [['2017', 'X', 1,     -5, 0,     1000,       45,             2],
                            ['2017', 'X', 1,     6, 2,     635,       825,             189],
                            ['2018', 'X', 1,    0,     1000,       45,             2,         123],
                            ['2018', 'X', 1,     2,     635,       825,             189, 849],
                            ['2019', 'X', 1, 1000,     45,         2,          123,         5],
                            ['2019', 'X', 1,     635,       825,             189, 849, 274],
                            ['2018', 'Y', 2, -32, 100,     65,         52,          54  ],
                            ['2019', 'Y', 2,   100,     65,         52,          54,         543]])
df.set_index('Year', inplace = True)


miner = data_patterns.PatternMiner(df)

print(miner.df_patterns)

# new_df = new_df.reset_index()
# new_df['2018'] = new_df['2018'].shift(-1)
# new_df['C'] = new_df['Name'] == new_df['Name'].shift(1).fillna(new_df['Name'])
# new_df['C'] = new_df['C'].shift(-1)
# new_df = new_df.fillna(0)
# new_df = new_df[new_df['C']==True]
# del new_df['C']
# new_df.set_index(['Name','Datapoint'], inplace = True)
# print(new_df)
p1 = {'name'     : 'Pattern 1',
         'pattern'  : '=',
         'parameters': {'shift': 1}}
df_patterns = miner.find(p1)

print(df_patterns.to_string())
