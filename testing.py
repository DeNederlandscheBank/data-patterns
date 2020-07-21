import data_patterns.data_patterns
import numpy as np
import pandas as pd
df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
          data   = [['Insurer  1', 'life & insurerd',     1000,     800,       0,             200,         200],
                    ['Insurer  2', 'non-life & insurer', 4000,     0,         3200,          800,         800],
                    ['Insurer  2', 'non-life & insurer', 800,      0,         700,           100,         100],
                    ['Insurer  1', 'life & insurer',     2500,     1800,      0,             700,         700],
                    ['Insurer  2', 'non-life & insurer', 2100,     0,         2200,          200,         200],
                    ['Insurer  1', 'life & insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'life & insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  1', 'life & insurer',     9000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurerd', 9000,     8800,      0,             200,         200],
                    ['Insurer  2', 'non-life & insurer', 9000,     0,         8800,          200,         199.99]])

df['LA'] = 0

# df.set_index(['newn', 'gg'], inplace=True)
print(df)
# df['New_ass'] = df['Assets']
# df = df.drop(['Assets'], 1)
# p1 ={'name'      : 'sum pattern',
#                           'pattern'   : '-->',
#                           'P_columns' :['Name'],
#                           'Q_columns' :['Type', 'LA'],
#                           'parameters': {"min_confidence": 'highest',
#                                          "min_support"   : 1 }}
# miner = data_patterns.PatternMiner(df)
# df_patterns = miner.find(p1)
#
# print(miner.df_data)
# print(df_patterns.loc[0,'pattern_def'])
#
# print(miner.df_patterns)
# print(miner.metapatterns)
#
# df_ana = miner.analyze()
# print(miner.df_results)
#
# df_cor = miner.correct_data()
# print(df_cor)
#






#
pattern ={'name'      : 'sum pattern',
                          'expression'   : 'IF {"Assets"} <> 0|{"TV-life"}<>0 THEN {"Excess"}<>0',
                          'parameters': {"min_confidence": 0.2,
                                         "min_support"   : 1}}
miner = data_patterns.PatternMiner(df)
df_patterns = miner.find(pattern)
print(df_patterns.to_string())
print(df_patterns.loc[0,'pandas co'])
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
print(data_patterns.get_value('IF {"S.26.04.01.05,R1200,C0230"} !=0 THEN {"S.26.04.01.05,R1200,C0230"} = MAX({"S.26.04.01.05,R1200,C0190"} - {"S.26.04.01.05,R1200,C0200"} - {"S.26.04.01.05,R1200,C0210"} + {"S.26.04.01.05,R1200,C0220"},0)', 2,1))
print(data_patterns.get_value('{"S.26.05.01.02,R0300,C0100"}>({"S.26.05.01.01,R0230,C0020"}*3*{"S.26.05.01.01,R0220,C0090"})', 1,1))
