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
                            ['Insurer  6', "life insurer",     9000,     8800,      0,             200,         201],
                            ['Insurer  7', 'life insurer',     9000,     6,         8800,          200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     0,         8800,          200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
# df.set_index('Name', inplace = True)
df['LA'] = np.nan
df['LB'] = np.nan
df['LA'].iloc[3] = 1
df['LB'].iloc[3] = 1
print(df)
p1 = {'name'     : 'Pattern 1','expression':'IF {"Assets"} > 0 THEN {"Excess"} > 0','parameters':{'min_support':0,'min_confidence':0,'decimal':0, 'expres':True, 'notNaN' :True} }
miner = data_patterns.PatternMiner(df)

df_patterns = miner.find(p1)
miner.df_data['Assets'] = 0
df_ana = miner.analyze()
print(miner.df_patterns)
print(miner.df_results)
