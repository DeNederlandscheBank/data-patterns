'''Encoding definitions
'''

#imports
import numpy as np
import pandas as pd

__author__ = """De Nederlandsche Bank"""
__email__ = 'ECDB_berichten@dnb.nl'
__version__ = '0.1.6'

def percentage(c):
    encoded = ["= 100%" if i == 1 else 
               "= 50%"  if i == 0.5 else
               "= 0%"   if i == 0 else
               "< 50%"  if i < 0.5 and i != 0 else
               "> 50%"  if i > 0.5 and i != 1 else "unknown" for i in c]
    return encoded

def reported(c):
    if not isinstance(c, pd.Series):
        encoded = ["not reported" if np.isnan(i) else "not reported" if i == 0 else "reported" for i in c]
    else:    
        encoded = pd.Series(index = c.index, data = ["not reported" if np.isnan(i) else "not reported" if i == 0 else "reported" for i in c])
    return encoded

def nonreported(c):
    encoded = ["not reported" if pd.isna(i) else str(i) for i in c]
    return encoded

EEA_countries = ['AUSTRIA',
                 'BELGIUM',
                 'BULGARIA',
                 'CROATIA',
                 'CYPRUS',
                 'CZECH REPUBLIC',
                 'CZECHIA',
                 'DENMARK',
                 'ESTONIA',
                 'FINLAND',
                 'FRANCE',
                 'GERMANY',
                 'GREECE',
                 'HUNGARY',
                 'ICELAND',
                 'IRELAND',
                 'ITALY',
                 'LATVIA',
                 'LIECHTENSTEIN',
                 'LITHUANIA',
                 'LUXEMBOURG',
                 'MALTA',
                 'NETHERLANDS',
                 'NORWAY',
                 'POLAND',
                 'PORTUGAL',
                 'ROMANIA',
                 'SLOVAKIA',
                 'SPAIN',
                 'SWEDEN',
                 'SWITZERLAND',
                 'UNITED KINGDOM']

def EEA_country(iterable):
    return ["EEA" if (i in EEA_countries) else "non-EEA" for i in iterable]

ENCODINGS_DICT = {'percentage' : percentage,
                  'reported'   : reported,
                  'nonreported': nonreported,
                  'EEA_country': EEA_country}
