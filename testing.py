import data_patterns
import numpy as np
import pandas as pd
import logging
import os

# logging.basicConfig()
# logging.getLogger().setLevel(logging.WARNING)

df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 0,     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurerx', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insu"rer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', "lif'e insurer",     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     6,         8800,          200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     0,         8800,          200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
df.set_index('Name', inplace = True)
# df['LA'] = None
# df['LB'] = None
# p1 = {'name'     : 'Pattern 1', 'expression':'ABS({"LB"} - {"LA"}) <= ABS({"LA"})*0.1', 'parameters':{'min_conf':0,'expres':True}}
# miner = data_patterns.PatternMiner(df)
#
# df_patterns = miner.find(p1)
# print(df_patterns)

result = data_patterns.load_overzicht('../','test_overzicht.xlsx')
print(result['df'][0]['parameters']['nonzero'])
