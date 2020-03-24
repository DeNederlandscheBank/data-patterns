# -*- coding: utf-8 -*-

"""Main module."""

# imports
import pandas as pd
import numpy as np
import copy
import xlsxwriter
import operator
import re
import ast
from functools import reduce
import itertools
import logging
from .constants import *
from .transform import *
from .encodings import *
#import optimized

__author__ = """De Nederlandsche Bank"""
__email__ = 'ECDB_berichten@dnb.nl'
__version__ = '0.1.13'

class PatternMiner:

    '''
    A PatternMiner object mines patterns in a Pandas DataFrame.

    Parameters
    ----------
    dataframe : DataFrame, optional, the dataframe with data used for training and testing (optional)
    metapatterns : list of dictionaries, optional

    Attributes
    ----------

    dataframe : Dataframe, shape (n_observations,)
        Dataframe with most recent data used for training and testing

    metapatterns : list of dictionaries (optional)
        a metapattern is a dict with
            'name': identifier of the metapattern (optional)
            'P_columns': columns of dataframe (P part of metapattern)
            'Q_columns': columns of datafrane (Q part of metapattern)
            'parameters': minimum confidence, patterns with higher confidence are included (optional)
            'encode': encoding definitions of the columns (optional)

    data : list, shape (n_patterns,)
        Patterns with statistics and confirmation and exceptions

    Examples
    --------

    See Also
    --------

    Notes
    -----

    '''

    def __init__(self, *args, **kwargs):
        self.df_data = None
        self.df_patterns = None
        self.metapatterns = None
        self.__process_parameters(*args, **kwargs)

    def find(self, *args, **kwargs):
        '''General function to find patterns
        '''
        self.__process_parameters(*args, **kwargs)

        assert self.metapatterns is not None, "No patterns defined."
        assert self.df_data is not None, "No dataframe defined."

        new_df_patterns = derive_patterns(**kwargs, metapatterns = self.metapatterns, dataframe = self.df_data)

        if (not kwargs.get('append', False)) or (self.df_patterns is None):
            self.df_patterns = new_df_patterns
        else:
            if len(new_df_patterns.index) > 0:
                self.df_patterns.append(new_df_patterns)

        return self.df_patterns

    def analyze(self, *args, **kwargs):
        '''General function to analyze data given a list of patterns
        '''
        self.__process_parameters(*args, **kwargs)

        assert self.df_patterns is not None, "No patterns defined."
        assert self.df_data is not None, "No data defined."

        self.df_patterns = update_statistics(dataframe = self.df_data, df_patterns = self.df_patterns)

        df_results = derive_results(**kwargs, df_patterns = self.df_patterns, dataframe = self.df_data)

        return df_results

    def update_statistics(self, *args, **kwargs):
        '''Function that updates the pattern statistics in df_patterns
        '''
        self.__process_parameters(*args, **kwargs)

        assert self.df_patterns is not None, "No patterns defined."
        assert self.df_data is not None, "No data defined."

        self.df_patterns = update_statistics(dataframe = self.df_data, df_patterns = self.df_patterns)

        return self.df_patterns

    def convert_labels(self, df1, df2):
        ''' converts the column names of a pattern dataframe
        '''
        return to_dataframe(patterns = convert_columns(self.df_patterns, df1, df2))

    def __process_parameters(self, *args, **kwargs):
        '''Update variables in the object
        '''
        self.metapatterns = self.__process_key('metapatterns', dict, self.metapatterns, *args, **kwargs)
        self.metapatterns = self.__process_key('metapatterns', list, self.metapatterns, *args, **kwargs)
        self.df_patterns = self.__process_key('df_patterns', None, self.df_patterns, *args, **kwargs)
        self.df_data = self.__process_key('dataframe', pd.DataFrame, self.df_data, *args, **kwargs)

        if isinstance(self.metapatterns, dict):
            self.metapatterns = [self.metapatterns]

        return None

    def __process_key(self, key, key_type, current, *args, **kwargs):
        '''
        '''
        if key in kwargs.keys():
            return kwargs.pop(key)
        else:
            for arg in args:
                if (key_type is not None) and isinstance(arg, key_type):
                      return arg
        return current

def derive_patterns(dataframe   = None,
                    metapatterns = None):
    '''Derive patterns from metapatterns
       In two flavours: association rules (-->) and quantitative rules (other)
    '''
    df_patterns = pd.DataFrame(columns = PATTERNS_COLUMNS)
    for metapattern in metapatterns:
        if metapattern.get("pattern", "-->") == "-->":
            patterns = derive_association_patterns(metapattern = metapattern,
                                                   dataframe = dataframe)
        else:
            patterns = derive_quantitative_patterns(metapattern = metapattern,
                                                    dataframe = dataframe)
        df_patterns = df_patterns.append(patterns, ignore_index = True)
    df_patterns[CLUSTER] = df_patterns[CLUSTER].astype(np.int64)
    df_patterns[SUPPORT] = df_patterns[SUPPORT].astype(np.int64)
    df_patterns[EXCEPTIONS] = df_patterns[EXCEPTIONS].astype(np.int64)
    df_patterns.index.name = 'index'
    return PatternDataFrame(df_patterns)

def derive_association_patterns(metapattern = None,
                                dataframe = None):
    '''Derive association rule patterns
       If no columns are given, then the algorithm searches for all possibilities
    '''
    new_list = list()

    parameters = metapattern.get("parameters", {})

    include_subsets = parameters.get("include_subsets", False)
    include_subsets_p = parameters.get("include_subsets_P_columns", False)
    include_subsets_q = parameters.get("include_subsets_Q_columns", False)

    if include_subsets:
        p_part = None
        q_part = None
        col_total_p = metapattern.get("P_columns", None)
        col_total_q = metapattern.get("Q_columns", None)
    elif include_subsets_q:
        q_part = None
        col_total_q = metapattern.get("Q_columns", None)
        p_part = metapattern.get("P_columns", None)
        col_total_p = dataframe.columns
    elif include_subsets_p:
        p_part = None
        col_total_p = metapattern.get("P_columns", None)
        q_part = metapattern.get("Q_columns", None)
        col_total_q = dataframe.columns
    else:
        p_part = metapattern.get("P_columns", None)
        q_part = metapattern.get("Q_columns", None)
        col_total_p = dataframe.columns
        col_total_q = dataframe.columns

    metapattern = copy.deepcopy(metapattern)

    # there are four cases:
    # - p and q are given,
    # - p is given but q is not given,
    # - q is given but p is not,
    # - p and q are not given
    if ((p_part is None) and (q_part is not None)):
        p_set = [col for col in col_total_p if col not in metapattern["Q_columns"]]
        p_set = itertools.chain.from_iterable(itertools.combinations(p_set, n+1) for n in range(len(p_set)))
        for item in p_set:
            metapattern["P_columns"] = list(item)
            new_patterns = derive_patterns_from_metapattern(metapattern = metapattern, dataframe = dataframe)
            new_list.extend(new_patterns)
    elif ((q_part is None) and (p_part is not None)):
        q_set = [col for col in col_total_q if col not in metapattern["P_columns"]]
        q_set = itertools.chain.from_iterable(itertools.combinations(q_set, n+1) for n in range(len(q_set)))
        for item in q_set:
            metapattern["Q_columns"] = list(item)
            new_patterns = derive_patterns_from_metapattern(metapattern = metapattern, dataframe = dataframe)
            new_list.extend(new_patterns)
    elif ((q_part is None) and (p_part is None)):
        p_set = [col for col in col_total_p]
        p_set = itertools.chain.from_iterable(itertools.combinations(p_set, n+1) for n in range(len(p_set)))
        for p_item in p_set:
            q_set = [col for col in col_total_q if col not in p_item]
            q_set = itertools.chain.from_iterable(itertools.combinations(q_set, n+1) for n in range(len(q_set)))
            for q_item in q_set:
                metapattern["Q_columns"] = list(q_item)
                metapattern["P_columns"] = list(p_item)
                new_patterns = derive_patterns_from_metapattern(metapattern = metapattern, dataframe = dataframe)
                new_list.extend(new_patterns)
    else:
        new_patterns = derive_patterns_from_metapattern(metapattern = metapattern, dataframe = dataframe)
        new_list.extend(new_patterns)
    df_patterns = to_dataframe(patterns = new_list, parameters = parameters)
    return df_patterns

def to_dataframe(patterns = None, parameters = {}):
    '''Convert list of patterns to dataframe with patterns
    '''
    # unpack pattern_id and pattern and patterns_stats and exclude co and ex and set pattern status to unknown
    patterns = list(patterns)
    if len(patterns) > 0:
        data = [pattern_id + pattern + pattern_stats + [INITIAL_PATTERN_STATUS] + [{}] +
               [pandas_co, pandas_ex, xbrl_co, xbrl_ex] for [pattern_id, pattern, pattern_stats, pandas_co, pandas_ex, xbrl_co, xbrl_ex] in patterns]
        df = pd.DataFrame(data = data, columns = PATTERNS_COLUMNS)
        df.index.name = 'index'
    else:
        df = pd.DataFrame(columns = PATTERNS_COLUMNS)
        df.index.name = 'index'
    return df

def update_statistics(dataframe = None,
                      df_patterns = None):
    '''Update statistics in df_patterns with statistics from the data by evaluating pandas expressions
    '''
    encodings = get_encodings()
    df_new_patterns = pd.DataFrame()
    if (dataframe is not None) and (df_patterns is not None):
        # adding the levels of the index to the columns (so they can be used for finding rules)
        for level in range(len(dataframe.index.names)):
            dataframe[dataframe.index.names[level]] = dataframe.index.get_level_values(level = level)
        for idx in df_patterns.index:
            # Calculate pattern statistics (from evaluating pandas expressions)
            pandas_co = df_patterns.loc[idx, PANDAS_CO]
            pandas_ex = df_patterns.loc[idx, PANDAS_EX]
            n_co = len(eval(pandas_co, encodings, {'df': dataframe}).index)
            n_ex = len(eval(pandas_ex, encodings, {'df': dataframe}).index)
            total = n_co + n_ex
            if total > 0:
                conf = np.round(n_co / total, 4)
            else:
                conf = 0
            df_patterns.loc[idx, SUPPORT] = n_co
            df_patterns.loc[idx, EXCEPTIONS] = n_ex
            df_patterns.loc[idx, CONFIDENCE] = conf
            df_new_patterns = df_patterns
        # deleting the levels of the index to the columns
        for level in range(len(dataframe.index.names)):
            del dataframe[dataframe.index.names[level]]
    return df_new_patterns

def get_encodings():
    for item in encodings_definitions:
        exec(encodings_definitions[item])
    encodings = {}
    for item in encodings_definitions.keys():
        encodings[item]= locals()[item]
    return encodings

def derive_results(dataframe = None,
                   P_dataframe = None,
                   Q_dataframe = None,
                   df_patterns = None):
    '''Results (patterns applied to data) are derived
       All info of the patterns is included in the results
    '''
    if (P_dataframe is not None) and (Q_dataframe is not None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            print("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []

    encodings = get_encodings()

    if (dataframe is not None) and (df_patterns is not None):
        df = dataframe.copy()
        results = list()
        for idx in df_patterns.index:
            if df_patterns.loc[idx, RELATION_TYPE] != "almost =":
                pandas_ex = df_patterns.loc[idx, PANDAS_EX]
                pandas_co = df_patterns.loc[idx, PANDAS_CO]
                results_ex = eval(pandas_ex, encodings, {'df': df}).index.values.tolist()
                results_co = eval(pandas_co, encodings, {'df': df}).index.values.tolist()
                for i in results_ex:
                    values_p = df.loc[i, df_patterns.loc[idx, P_COLUMNS]].values.tolist()
                    if type(df_patterns.loc[idx, Q_COLUMNS])==list:
                        values_q = df.loc[i, df_patterns.loc[idx, Q_COLUMNS]].values.tolist()
                    else:
                        values_q = df_patterns.loc[idx, Q_COLUMNS]
                    results.append([False,
                                    df_patterns.loc[idx, "pattern_id"],
                                    df_patterns.loc[idx, "cluster"],
                                    i,
                                    df_patterns.loc[idx, "support"],
                                    df_patterns.loc[idx, "exceptions"],
                                    df_patterns.loc[idx, "confidence"],
                                    df_patterns.loc[idx, P_COLUMNS],
                                    df_patterns.loc[idx, "relation type"],
                                    df_patterns.loc[idx, Q_COLUMNS],
                                    df_patterns.loc[idx, "P"],
                                    df_patterns.loc[idx, "relation"],
                                    df_patterns.loc[idx, "Q"],
                                    values_p,
                                    values_q])
                for i in results_co:
                    values_p = df.loc[i, df_patterns.loc[idx, P_COLUMNS]].values.tolist()
                    if type(df_patterns.loc[idx, Q_COLUMNS])==list:
                        values_q = df.loc[i, df_patterns.loc[idx, Q_COLUMNS]].values.tolist()
                    else:
                        values_q = df_patterns.loc[idx, Q_COLUMNS]
                    results.append([True,
                                    df_patterns.loc[idx, "pattern_id"],
                                    df_patterns.loc[idx, "cluster"],
                                    i,
                                    df_patterns.loc[idx, "support"],
                                    df_patterns.loc[idx, "exceptions"],
                                    df_patterns.loc[idx, "confidence"],
                                    df_patterns.loc[idx, P_COLUMNS],
                                    df_patterns.loc[idx, "relation type"],
                                    df_patterns.loc[idx, Q_COLUMNS],
                                    df_patterns.loc[idx, "P"],
                                    df_patterns.loc[idx, "relation"],
                                    df_patterns.loc[idx, "Q"],
                                    values_p,
                                    values_q])
        df_results = pd.DataFrame(data = results, columns = RESULTS_COLUMNS)
        df_results.sort_values(by = ["index", "confidence", "support"], ascending = [True, False, False], inplace = True)
        df_results.set_index(["index"], inplace = True)
        try:
            df_results.index = pd.MultiIndex.from_tuples(df_results.index)
        except:
            df_results.index = df_results.index
    return ResultDataFrame(df_results)

def derive_patterns_from_metapattern(dataframe = None,
                                     metapattern = None):
    '''Here we derive the patterns from the metapattern definitions
       by evaluating the pandas expressions of all potential patterns
    '''
    # get items from metapattern definition
    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")
    encode = metapattern.get(ENCODE, {})
    P_columns = sorted(metapattern["P_columns"])
    Q_columns = sorted(metapattern["Q_columns"])
    confidence, support = get_parameters(parameters)
    # adding index levels to columns (in case the pattern contains index elements)
    for level in range(len(dataframe.index.names)):
        dataframe[dataframe.index.names[level]] = dataframe.index.get_level_values(level = level)
    # derive df_feature list from P and Q (we use a copy, so we can change values for encodings)
    df_features = dataframe[P_columns + Q_columns].copy()
    # execute dynamic encoding functions
    encodings = get_encodings()
    # perform encodings on df_features
    if encode != {}:
        for c in df_features.columns:
            if c in encode.keys():
                df_features[c] = eval(str(encode[c])+ "(s)", encodings, {'s': df_features[c]})
    df_potential_patterns = df_features.drop_duplicates() # these are all unique combinations, i.e. the potential rules
    patterns = list()
    for idx in range(len(df_potential_patterns.index)):
        P_values = list(df_potential_patterns[P_columns].values[idx])
        Q_values = list(df_potential_patterns[Q_columns].values[idx])
        pattern = [P_columns, "-->", Q_columns, P_values, "-->", Q_values]
        pandas_co = to_pandas_expression(pattern, encode, True, parameters)
        pandas_ex = to_pandas_expression(pattern, encode, False, parameters)
        n_co = len(eval(pandas_co, encodings, {'df': dataframe}).index)
        n_ex = len(eval(pandas_ex, encodings, {'df': dataframe}).index)
        conf = np.round(n_co / (n_co + n_ex), 4)
        if ((conf >= confidence) and (n_co >= support)):
            xbrl_co = to_xbrl_expression(pattern, encode, True, parameters)
            xbrl_ex = to_xbrl_expression(pattern, encode, False, parameters)
            patterns.append([[name, 0], pattern, [n_co, n_ex, conf], pandas_co, pandas_ex, xbrl_co, xbrl_ex])
    # deleting the levels of the index to the columns
    for level in range(len(dataframe.index.names)):
        del dataframe[dataframe.index.names[level]]
    return patterns

# def convert_columns(patterns = [],
#                     dataframe_input = None,
#                     dataframe_output = None):
#     '''
#     '''
#     new_patterns = list()
#     for pattern_id, pattern, pattern_stats, encode in patterns:
#         new_pattern = [ [dataframe_output.columns[dataframe_input.columns.get_loc(item)] for item in pattern[0]],
#                         pattern[1],
#                         [dataframe_output.columns[dataframe_input.columns.get_loc(item)] for item in pattern[2]]] + pattern[3:6]
#         new_encode = {dataframe_output.columns[dataframe_input.columns.get_loc(item)]: encode[item] for item in encode.keys()}
#         new_patterns.append([pattern_id, new_pattern, pattern_stats, new_encode])
#     return new_patterns

def logical_equivalence(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ((nonzero_c1 & nonzero_c2) | (~nonzero_c1 & ~nonzero_c2))

# implication
def logical_implication(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ~(nonzero_c1 & ~nonzero_c2)

operators = {'>' : operator.gt,
             '<' : operator.lt,
             '>=': operator.ge,
             '<=': operator.le,
             '=' : operator.eq,
             '!=': operator.ne,
             '<->': logical_equivalence,
             '-->': logical_implication}

preprocess = {'>':   operator.and_,
              '<':   operator.and_,
              '>=':  operator.and_,
              '<=':  operator.and_,
              '=' :  operator.and_,
              '!=':  operator.and_,
              'sum': operator.and_,
              'ratio': operator.and_,
              '<->': operator.or_,
              '-->': operator.or_}

logicals = {
        'and': operator.and_,
        'or': operator.or_,
        'xor':operator.xor
}

operations = {
 '+' : operator.add,
 '-' : operator.sub,
 '*' : operator.mul,
 '/' : operator.truediv
}
def derive_pattern_statistics(co):
    # co_sum is the support of the pattern
    co_sum = co.sum()
    #co_sum = optimized.apply_sum(co)
    ex_sum = len(co) - co_sum
    # conf is the confidence of the pattern
    conf = np.round(co_sum / (co_sum + ex_sum), 4)
    # oddsratio is a correlation measure
    #oddsratio = (1 + co_sum) / (1 + ex_sum)
    return co_sum, ex_sum, conf #, oddsratio

def derive_pattern_data(df,
                        P_columns,
                        Q_columns,
                        pattern,
                        name,
                        co,
                        confidence,
                        data_filter):
    '''
    '''
    data = list()
    # pattern statistics
    co_sum, ex_sum, conf = derive_pattern_statistics(co)
    # we only store the patterns with confidence higher than conf
    if conf >= confidence:
        data = [[name, 0], [P_columns, pattern, Q_columns, '', '', ''], [co_sum, ex_sum, conf]]
    return data

def get_parameters(parameters):
    confidence = parameters.get("min_confidence", 0.75)
    support = parameters.get("min_support", 2)
    return confidence, support

def patterns_column_value(dataframe = None,
                          pattern   = None,
                          pattern_name = "value",
                          columns   = None,
                          value     = None,
                          parameters= {}):
    '''Generate patterns of the form [c1] operator value where c1 is in columns
    '''
    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T
    for c in columns:
        # confirmations and exceptions of the pattern, a list of booleans
        co = reduce(operators[pattern], [data_array[c, :], value])
        pattern_data = derive_pattern_data(dataframe,
                                           [dataframe.columns[c]],
                                           value,
                                           pattern,
                                           pattern_name,
                                           co,
                                           confidence,
                                           None)
        if pattern_data and len(co) >= support:
            yield pattern_data

def patterns_compare_columns_simple(dataframe = None,
                          pattern   = None,
                          pattern_name = "comparative",
                          P_columns  = None,
                          Q_columns  = None,
                          both_ways = False,
                          values     = None,
                          parameters= {}):
    '''
    Generate patterns where you compare one column c1
     to a value or other column and condition on other column c0
    '''
    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T
    columns = dataframe.columns
    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # compare to other columns
                if values[0] in columns and values[1] in columns:
                    co1 = reduce(operators[pattern[0]], [data_array[c0, :], dataframe[[values[0]]].values.T])[0]
                    co2 = reduce(operators[pattern[1]], [data_array[c1, :], dataframe[[values[1]]].values.T])[0]
                elif values[1] in columns:
                    co1 = reduce(operators[pattern[0]], [data_array[c0, :], values[0]])
                    co2 = reduce(operators[pattern[1]], [data_array[c1, :], dataframe[[values[1]]].values.T])[0]
                elif values[0] in columns:
                    co1 = reduce(operators[pattern[0]], [data_array[c0, :], dataframe[[values[0]]].values.T])[0]
                    co2 = reduce(operators[pattern[1]], [data_array[c1, :], values[1]])
                # if values are integer then it is a set value
                else:
                    co1 = reduce(operators[pattern[0]], [data_array[c0, :], values[0]])
                    co2 = reduce(operators[pattern[1]], [data_array[c1, :], values[1]])
                # If we want to work both ways then we set that value to True
                if both_ways:
                    co = np.logical_not(np.logical_xor(co1,co2))
                else:
                    co = np.logical_or(np.logical_not(co1),co2)
                co_sum, ex_sum, conf = derive_pattern_statistics(co)
                if (conf >= confidence) and (co_sum >= support):
                    pattern_id = [[dataframe.columns[c0]],
                               (pattern[0] + str(values[0]) + ', ' + pattern[1] +str(values[1])),
                               [dataframe.columns[c1]], '', '', '']
                    pattern_data = [[pattern_name, 0],
                                    pattern_id,
                                    [co_sum, ex_sum, conf]]
                    if pattern_data and len(co) >= support:
                        yield pattern_data

def patterns_compare_columns_complex(dataframe = None,
                          pattern   = None,
                          pattern_name = "comparative",
                          P_columns  = None,
                          Q_columns  = None,
                          P_logics = None, # Logical operators in the order that you wan tthem applied
                          Q_logics = None,
                          both_ways = False,
                          values     = None,
                          parameters= {}):
    '''    Generate patterns where you compare one column combination of columns with operators
         to a value or other column and condition on combination of other columns
    '''
    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T
    columns = dataframe.columns

    # Split values, patterns and lists for P and Q
    values_P = values[0]
    values_Q = values[1]
    patterns_P = pattern[0]
    patterns_Q = pattern[1]
    co1_list = []
    co2_list = []

    # make array of truth values for P and Q columns
    for i in range(len(values_P)):
        if values_P[i] in columns: # compare column with other column
            co1_list.append(reduce(operators[patterns_P[i]], [data_array[P_columns[i], :], dataframe[[values_P[i]]].values.T])[0])
        else: # compare with value
            co1_list.append(reduce(operators[patterns_P[i]], [data_array[P_columns[i], :], values_P[i]]))

    # Apply the logic operator on the truth tables in the right order
    co1 =co1_list[0]
    for i in range(1, len(co1_list)):
        co1 = reduce(logicals[P_logics[i-1]], [co1, co1_list[i]])

    # Do the same for Q
    for i in range(len(values_Q)):
        if values_Q[i] in columns:
            co2_list.append(reduce(operators[patterns_Q[i]], [data_array[Q_columns[i], :], dataframe[[values_Q[i]]].values.T])[0])
        else:
            co2_list.append(reduce(operators[patterns_Q[i]], [data_array[Q_columns[i], :], values_Q[i]]))

    co2 =co2_list[0]
    for i in range(1, len(co2_list)):
        co2 = reduce(logicals[Q_logics[i-1]], [co2, co2_list[i]])

    # Combine P and Q, if both_ways then it works both ways
    if both_ways:
        co = np.logical_not(np.logical_xor(co1,co2))
    else:
        co = np.logical_or(np.logical_not(co1),co2)

    # get results
    co_sum, ex_sum, conf = derive_pattern_statistics(co)
    if (conf >= confidence) and (co_sum >= support):
        pattern_id = [[dataframe.columns[[P_columns]]],
                   (str(pattern[0]) + str(values[0]) + ', ' + str(pattern[1]) +str(values[1])),
                   [dataframe.columns[[Q_columns]]], '', '', '']
        pattern_data = [[pattern_name, 0],
                        pattern_id,
                        [co_sum, ex_sum, conf]]
        if pattern_data and len(co) >= support:
            yield pattern_data

def patterns_column_column(dataframe  = None,
                           pattern    = None,
                           pattern_name = "column",
                           P_columns  = None,
                           Q_columns  = None,
                           parameters = {}):
    '''Generate patterns of the form [c1] operator [c2] where c1 and c2 are in columns
    '''
    confidence, support = get_parameters(parameters)
    decimal = parameters.get("decimal", 8)
    preprocess_operator = preprocess[pattern]
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = initial_data_array != 0
    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                if data_filter.any():
                    data_array = initial_data_array[:, data_filter]
                    if data_array.any():
                        # confirmations of the pattern, a list of booleans
                        if pattern == "=":
                            co = np.abs(data_array[c0, :] - data_array[c1, :]) < 1.5 * 10**(-decimal)
                        else:
                            co = reduce(operators[pattern], data_array[[c0, c1], :])
                        pattern_data = derive_pattern_data(dataframe,
                                            [dataframe.columns[c0]],
                                            [dataframe.columns[c1]],
                                            pattern,
                                            pattern_name,
                                            co,
                                            confidence,
                                            data_filter)
                        if pattern_data and len(co) >= support:
                            yield pattern_data

def patterns_sums_column(dataframe  = None,
                         pattern_name = None,
                         P_columns  = None,
                         Q_columns  = None,
                         parameters = {}):
    '''Generate patterns of the form sum [c1-list] = [c2] where c1-list is column list and c2 is column
    '''
    confidence, support = get_parameters(parameters)
    sum_elements = parameters.get("sum_elements", 2)
    decimal = parameters.get("decimal", 0)
    preprocess_operator = preprocess["sum"]
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    n = len(dataframe.columns)
    # setup matrix to speed up proces (under development)
    # matrix = np.ones(shape = (n, n), dtype = bool)
    # for c in itertools.combinations(range(n), 2):
    #     v = (data_array[c[1], :] <= data_array[c[0], :] + 1).any() # all is too strict
    #     matrix[c[0], c[1]] = v
    #     matrix[c[1], c[0]] = ~v
    # np.fill_diagonal(matrix, False)

    for lhs_elements in range(2, sum_elements + 1):
        for rhs_column in Q_columns:
            start_array = initial_data_array
            # minus righthandside is taken so we can use sum function for all columns
            start_array[rhs_column, :] = -start_array[rhs_column, :]
            lhs_column_list = [col for col in P_columns if (col != rhs_column)]
            for lhs_columns in itertools.combinations(lhs_column_list, lhs_elements):
                all_columns = lhs_columns + (rhs_column,)
                data_filter = np.logical_and.reduce(nonzero[all_columns, :])
                if data_filter.any():
                    data_array = start_array[:, data_filter]
                    co = (abs(np.sum(data_array[all_columns, :], axis = 0)) < 1.5 * 10**(-decimal))
                    co_sum, ex_sum, conf = derive_pattern_statistics(co)
                    # we only store the patterns that satisfy criteria
                    if (conf >= confidence) and (co_sum >= support):
                        pattern = [[dataframe.columns[c] for c in lhs_columns],
                                   "sum",
                                   [dataframe.columns[rhs_column]], '', '', '']
                        pattern_data = [[pattern_name, 0],
                                        pattern,
                                        [co_sum, ex_sum, conf]]
                        if pattern_data:
                            yield pattern_data

def patterns_ratio(dataframe  = None,
                   pattern_name = None,
                   P_columns  = None,
                   Q_columns  = None,
                   parameters = {}):
    """Generate patterns with ratios
    """
    confidence, support = get_parameters(parameters)
    limit_denominator = parameters.get("limit_denominator", 10000000)
    decimal = parameters.get("decimal", 6)
    preprocess_operator = preprocess["ratio"]
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                data_array = map(lambda e: Fraction(e).limit_denominator(limit_denominator),
                                 dataframe.values[data_filter, c0] / dataframe.values[data_filter, c1])
                ratios = pd.Series(data_array)
                if support >= 2:
                    possible_ratios = ratios.loc[ratios.duplicated(keep = False)].unique()
                else:
                    possible_ratios = ratios.unique()
                for v in possible_ratios:
                    if (abs(v) > 1.5 * 10**(-decimal)) and (v > -1) and (v < 1):
                        # confirmations of the pattern, a list of booleans
                        co = ratios==v
                        co_sum, ex_sum, conf = derive_pattern_statistics(co)
                        if (conf >= confidence) and (co_sum >= support):
                            pattern_data = [[pattern_name, 0],
                                            [dataframe.columns[c0],
                                             'ratio',
                                            [dataframe.columns[c1]], '', '', ''],
                                            [co_sum, ex_sum, conf], {}]
                            if pattern_data:
                                yield pattern_data

def derive_quantitative_patterns(metapattern  = None,
                                 dataframe    = None):
    '''
    '''
    logger = logging.getLogger("quant-pattern")

    P_dataframe = metapattern.get("P_dataframe", None)
    Q_dataframe = metapattern.get("Q_dataframe", None)
    pattern = metapattern.get("pattern", None)
    pattern_name = metapattern.get("name", None)
    columns = metapattern.get("columns", None)
    P_columns = metapattern.get("P_columns", None)
    Q_columns = metapattern.get("Q_columns", None)
    value = metapattern.get("value", None)
    parameters = metapattern.get("parameters", None)
    values = metapattern.get("values", None)
    P_logics = metapattern.get("P_logics", None)
    Q_logics = metapattern.get("Q_logics", None)
    both_ways = metapattern.get("both_ways", False)
    # if P_dataframe and Q_dataframe are given then join the dataframes and select columns
    if (P_dataframe is not None) and (Q_dataframe is not None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            logger.error("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []
        P_columns = P_dataframe.columns
        Q_columns = Q_dataframe.columns

    # select all columns with numerical values
    numerical_columns = [dataframe.columns[c] for c in range(len(dataframe.columns))
                            if ((dataframe.dtypes[c] == 'float64') or (dataframe.dtypes[c] == 'int64')) and (dataframe.iloc[:, c] != 0).any()]
    dataframe_num = dataframe[numerical_columns]

    if P_columns is not None:
        if values:
            P_columns = [dataframe.columns.get_loc(c) for c in P_columns]
        else:
            P_columns = [dataframe_num.columns.get_loc(c) for c in P_columns if c in numerical_columns]
    else:
        P_columns = range(len(dataframe_num.columns))

    if Q_columns is not None:
        if values:
            Q_columns = [dataframe.columns.get_loc(c) for c in Q_columns]
        else:
            Q_columns = [dataframe_num.columns.get_loc(c) for c in Q_columns if c in numerical_columns]
    else:
        Q_columns = range(len(dataframe_num.columns))

    if columns is not None:
        columns = [dataframe_num.columns.get_loc(c) for c in columns if c in numerical_columns]
    else:
        columns = range(len(dataframe_num.columns))

    logger.info('START: %s (%s) in P_columns = %s and Q_columns = %s', pattern, pattern_name, str(P_columns), str(Q_columns))

    # if a value is given -> columns pattern value
    if value is not None:
        results = patterns_column_value(dataframe = dataframe_num,
                                        pattern = pattern,
                                        pattern_name = pattern_name,
                                        columns = columns,
                                        value = value,
                                        parameters = parameters)
    elif pattern == 'sum':
        results = patterns_sums_column(dataframe = dataframe_num,
                                       pattern_name = pattern_name,
                                       P_columns = P_columns,
                                       Q_columns = Q_columns,
                                       parameters = parameters)
    elif pattern == 'ratio':
        results = patterns_ratio(dataframe = dataframe_num,
                                 pattern_name = pattern_name,
                                 P_columns = P_columns,
                                 Q_columns = Q_columns,
                                 parameters = parameters)
    elif values is not None:
        if P_logics is None and Q_logics is None:
            results = patterns_compare_columns_simple(dataframe = dataframe,
                                            pattern = pattern,
                                            pattern_name = pattern_name,
                                             P_columns = P_columns,
                                             Q_columns = Q_columns,
                                             both_ways = both_ways,
                                            values = values,
                                            parameters = parameters)
        else:
            results = patterns_compare_columns_complex(dataframe = dataframe,
                                            pattern = pattern,
                                            pattern_name = pattern_name,
                                             P_columns = P_columns,
                                             Q_columns = Q_columns,
                                             P_logics = P_logics,
                                             Q_logics = Q_logics,
                                             both_ways = both_ways,
                                            values = values,
                                            parameters = parameters)

    # everything else -> c1 pattern c2
    else:
        results = patterns_column_column(dataframe = dataframe_num,
                                         pattern = pattern,
                                         pattern_name = pattern_name,
                                         P_columns = P_columns,
                                         Q_columns = Q_columns,
                                         parameters = parameters)
    patterns = [[pattern_id] + [pattern] + [pattern_stats] +
                [to_pandas_expression(pattern, {}, True, parameters),
                 to_pandas_expression(pattern, {}, False, parameters),
                 to_xbrl_expression(pattern, {}, True, parameters),
                 to_xbrl_expression(pattern, {}, False, parameters)] for [pattern_id, pattern, pattern_stats] in results]
    df = to_dataframe(patterns = patterns, parameters = parameters)

    logger.info('END: %s (%s)', pattern, pattern_name)

    return df

def read_excel(filename = None,
               dataframe = None,
               sheet_name = 'Patterns'):
    df = pd.read_excel(filename, sheet_name = sheet_name)
    df.fillna('', inplace = True)
    df[RELATION_TYPE] = df[RELATION_TYPE].str[1:]
    patterns = list()
    for row in df.index:
        P_columns = evaluate_excel_string(df.loc[row, P_COLUMNS])
        Q_columns = evaluate_excel_string(df.loc[row, Q_COLUMNS])
        P = evaluate_excel_string(df.loc[row, P_PART])
        Q = evaluate_excel_string(df.loc[row, Q_PART])
        encode = ast.literal_eval(df.loc[row, ENCODINGS])
        pandas_co = df.loc[row, PANDAS_CO]
        pandas_ex = df.loc[row, PANDAS_EX]
        xbrl_co = df.loc[row, XBRL_CO]
        xbrl_ex = df.loc[row, XBRL_EX]
        patterns.append([[df.loc[row, PATTERN_ID], 0],
                         [P_columns, df.loc[row, RELATION_TYPE], Q_columns,
                         P, df.loc[row, RELATION], Q],
                         [0, 0, 0], pandas_co, pandas_ex, xbrl_co, xbrl_ex])
    df_patterns = to_dataframe(patterns = patterns, parameters = {})
    if dataframe is not None:
        df_patterns = update_statistics(dataframe = dataframe, df_patterns = df_patterns)
    return df_patterns

def evaluate_excel_string(s):
    if s != '':
        if type(s)==str:
            return ast.literal_eval(s)
        else:
            return s
    else:
        return s

def to_xbrl_expression(pattern, encode, result_type, parameters):
    '''Placeholder for XBRL expression
    '''
    column_P = pattern[0]
    column_Q = pattern[2]
    # the content of the column
    value_P = pattern[3]
    value_Q = pattern[5]

    if pattern[1] != '-->':
        if pattern[1]=="=":
            expr = 'abs({' + str(column_P[0]) + '}'
            for p_item in column_P[1:]:
                expr += '+ {' + str(p_item) + '}'
            if type(column_Q)==list:
                expr += '-{' + str(column_Q[0]) + '}) '
            else:
                expr += '-' + str(column_Q) + ' '

            if result_type == True:
                expr += '< 1.5e'+str(-parameters.get("decimal", 8))
            else:
                expr += '>= 1.5e'+str(-parameters.get("decimal", 8))
        elif pattern[1]=="sum":
            nonzero = '({' + str(column_P[0]) + '}!=0)'
            for p_item in column_P[1:]:
                nonzero = nonzero + ' & ({' + str(p_item) + '}!=0)'
            if type(column_Q)==list:
                for q_item in column_Q:
                    nonzero = nonzero + ' & ({' + str(q_item) + '}!=0)'
            else:
                nonzero = nonzero + ' & ({' + str(column_Q) + '}!=0)'
            expr = '(' + nonzero + ') & (abs(({' + str(column_P[0]) + '}'
            for p_item in column_P[1:]:
                expr += '+{' + str(p_item) + '}'
            if type(column_Q)==list:
                expr += ')-{"' + str(column_Q[0]) + '"}) '
            else:
                expr += ')-' + str(column_Q) + ' '
            if result_type == True:
                expr += '< 1.5e'+str(-parameters.get("decimal", 0))+')'
            else:
                expr += '>= 1.5e'+str(-parameters.get("decimal", 0))+')'
        else:
            if result_type == True:
                string_pattern = str(pattern[1])
            else:
                if pattern[1] == "<":
                    string_pattern = ">="
                elif pattern[1] == "<=":
                    string_pattern = ">"
                elif pattern[1] == ">=":
                    string_pattern = "<"
                elif pattern[1] == ">":
                    string_pattern = "<="
                else:
                    string_pattern = "UNKNOWN"
            expr = '{' + str(column_P[0]) + '}'
            for p_item in column_P[1:]:
                expr += '+ {' + str(p_item) + '}'
            if type(column_Q)==list:
                expr += " " + string_pattern + '{' + str(column_Q[0]) + '}'
            else:
                expr += " " + string_pattern + ' ' + str(column_Q) + ''
    else:
        # if condition
        condition_P = ""
        for idx, cond in enumerate(column_P):
            if type(value_P[idx]) == str:
                r_string = '"' + str(value_P[idx]) + '"'
            else:
                r_string = str(value_P[idx])
            if column_P[idx] in encode.keys():
                condition_P = condition_P + '('+ encode[column_P[idx]]+ '({' + str(column_P[idx]) + '})=' + r_string + ")"
            else:
                condition_P = condition_P + '({' + str(column_P[idx]) + '}=' + r_string + ")"
            if cond != column_P[-1]:
                condition_P = condition_P + ' & '
        condition_Q = ""
        for idx, cond in enumerate(column_Q):
            if type(value_Q[idx]) == str:
                r_string = '"' + str(value_Q[idx]) + '"'
            else:
                r_string = str(value_Q[idx])
            if column_Q[idx] in encode.keys():
                condition_Q = condition_Q + '('+ encode[column_Q[idx]]+ '({' + str(column_Q[idx]) + '})=' + r_string + ")"
            else:
                condition_Q = condition_Q + '({' + str(column_Q[idx]) + '}=' + r_string + ")"
            if cond != column_Q[-1]:
                condition_Q = condition_Q + ' & '
        if result_type == False:
            expr = "IF " + condition_P + " THEN ~(" + condition_Q + ")"
        else:
            expr = "IF " + condition_P + " THEN (" + condition_Q + ")"
    return expr

def to_pandas_expression(pattern, encode, result_type, parameters):
    '''
    '''
    # the column name
    column_P = pattern[0]
    column_Q = pattern[2]
    # the content of the column
    value_P = pattern[3]
    value_Q = pattern[5]

    if pattern[1] != '-->':
        if pattern[1]=="=":
            expr = 'df[(abs((df["' + str(column_P[0]) + '"]'
            for p_item in column_P[1:]:
                expr += '+ df["' + str(p_item) + '"]'
            if type(column_Q)==list:
                expr += ')-df["' + str(column_Q[0]) + '"]) '
            else:
                expr += ')-' + str(column_Q) + ' '

            if result_type == True:
                expr += '< 1.5e'+str(-parameters.get("decimal", 8))+')]'
            else:
                expr += '>= 1.5e'+str(-parameters.get("decimal", 8))+')]'
        elif pattern[1]=="sum":
            nonzero = '(df["' + str(column_P[0]) + '"]!=0)'
            for p_item in column_P[1:]:
                nonzero = nonzero + ' & (df["' + str(p_item) + '"]!=0)'
            if type(column_Q)==list:
                for q_item in column_Q:
                    nonzero = nonzero + ' & (df["' + str(q_item) + '"]!=0)'
            else:
                nonzero = nonzero + ' & (df[' + str(column_Q) + ']!=0)'

            expr = 'df[(' + nonzero + ') & (abs((df["' + str(column_P[0]) + '"]'
            for p_item in column_P[1:]:
                expr += '+df["' + str(p_item) + '"]'
            if type(column_Q)==list:
                expr += ')-df["' + str(column_Q[0]) + '"]) '
            else:
                expr += ')-' + str(column_Q) + ' '
            if result_type == True:
                expr += '< 1.5e'+str(-parameters.get("decimal", 0))+')]'
            else:
                expr += '>= 1.5e'+str(-parameters.get("decimal", 0))+')]'
        else:
            if result_type == True:
                string_pattern = str(pattern[1])
            else:
                if pattern[1] == "<":
                    string_pattern = ">="
                elif pattern[1] == "<=":
                    string_pattern = ">"
                elif pattern[1] == ">=":
                    string_pattern = "<"
                elif pattern[1] == ">":
                    string_pattern = "<="
                else:
                    string_pattern = "UKNOWN"
            expr = 'df[(df["' + str(column_P[0]) + '"]'
            for p_item in column_P[1:]:
                expr += '+ df["' + str(p_item) + '"]'
            if type(column_Q)==list:
                expr += ")" + string_pattern + 'df["' + str(column_Q[0]) + '"]]'
            else:
                expr += ")" + string_pattern + ' ' + str(column_Q) + ']'
    else:
        # if condition
        condition_P = ""
        for idx, cond in enumerate(column_P):
            equal_str = "=="
            if type(value_P[idx]) == str:
                r_string = '"' + str(value_P[idx]) + '"'
            else:
                r_string = str(value_P[idx])
            if r_string == 'nan':
                r_string = 'isnull()'
                equal_str = "."


            if column_P[idx] in encode.keys():
                condition_P = condition_P + '('+ encode[column_P[idx]]+ '(df["' + str(column_P[idx]) + '"])'+ equal_str + r_string + ")"
            else:
                condition_P = condition_P + '(df["' + str(column_P[idx]) + '"]' + equal_str + r_string + ")"

            if cond != column_P[-1]:
                condition_P = condition_P + ' & '

        condition_Q = ""
        for idx, cond in enumerate(column_Q):
            equal_str = "=="
            if type(value_Q[idx]) == str:
                r_string = '"' + str(value_Q[idx]) + '"'
            else:
                r_string = str(value_Q[idx])
            if r_string == 'nan':
                r_string = 'isnull()'
                equal_str = "."

            if column_Q[idx] in encode.keys():
                condition_Q = condition_Q + '('+ encode[column_Q[idx]]+ '(df["' + str(column_Q[idx]) + '"])' + equal_str + r_string + ")"
            else:
                condition_Q = condition_Q + '(df["' + str(column_Q[idx]) + '"]' + equal_str + r_string + ")"

            if cond != column_Q[-1]:
                condition_Q = condition_Q + ' & '

        if result_type == False:
            expr = "df[" + condition_P + " & ~(" + condition_Q + ")]"
        else:
            expr = "df[" + condition_P + " & (" + condition_Q + ")]"

    return expr

def find_redundant_patterns(df_patterns = None):
    '''This function checks whether there are redundant patterns and changes pattern status accordingly
    so if [A, B, C] -> [Z] has conf = 0.95 and support = 10 and
          [A, B] -> [Z] has equal or better statistics then the former pattern is redundant
    '''
    for row in df_patterns.index:
        p_columns = df_patterns.loc[row, P_COLUMNS]
        q_columns = df_patterns.loc[row, Q_COLUMNS]
        p_items = df_patterns.loc[row, 'P']
        if len(p_columns) > 2: # only
            # determine all possible subsets of P and check whether they are better
            p_subsets = list(itertools.combinations(p_columns, len(p_columns) - 1))
            for subset in p_subsets:
                P_dict = {col: p_items[idx] for idx, col in enumerate(subset)}
                for i, row2 in enumerate(df_patterns.index):
                    p_columns2 = df_patterns.loc[row2, P_COLUMNS]
                    q_columns2 = df_patterns.loc[row2, Q_COLUMNS]
                    p_item2 = df_patterns.loc[row2, 'P']
                    if (set(q_columns2) == set(q_columns)) and (len(p_columns2) == len(P_dict.keys())):
                        equal = True
                        for key in P_dict.keys():
                            if key not in p_columns2:
                                equal = False
                            else:
                                if P_dict[key] not in p_item2:
                                    equal = False
                                else:
                                    if P_dict[key] != p_item2[p_item2.index(P_dict[key])]:
                                        equal = False
                        if equal:
                            if (df_patterns.loc[row, 'confidence'] <= df_patterns.loc[row2, 'confidence']) and (df_patterns.loc[row, 'support'] <= df_patterns.loc[row2, 'support']):
                                df_patterns.loc[row, 'pattern status'] = "redundant with pattern " + str(row2)
    return df_patterns

def make_new_columns(dataframe, columns, operation, new_names):
    df = dataframe
    # Make all new columns
    for i in range(len(new_names)):
        # amount of operations per new column
        for j in range(len(operation[i])):
            # apply the logical operators on the column step by step
            if j == 0:
                df[new_names[i]] = df.apply(lambda x: operations[operation[i][j]](x[columns[i][j]], x[columns[i][j+1]]), axis=1) # operation is a dict of possible operators
            else:
                df[new_names[i]] = df.apply(lambda x: operations[operation[i][j]](x[new_names[i]], x[columns[i][j+1]]), axis=1)

    return df
