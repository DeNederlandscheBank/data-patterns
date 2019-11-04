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
from .constants import *
from .transform import *
from .encodings import * 

__author__ = """De Nederlandsche Bank"""
__email__ = 'ECDB_berichten@dnb.nl'
__version__ = '0.1.0'

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
            'name': identifier of the metapattern
            'P': columns of dataframe (P part of metapattern)
            'Q': columns of datafrane (Q part of metapattern)
            'threshold': minimum confidence, patterns with higher confidence are included
            'encode': encoding definitions of the columns (optional)

    data : list, shape (n_patterns,)
        Patterns with statistics and confirmation and exceptions

    Examples
    --------
    >>> import patterns
    >>> df = pd.DataFrame((data = [[2, 3, 2.01, 0.1], 
                                   [2, 5, 2,    0.1], 
                                   [1, 3, 1,    0.2], 
                                   [1, 3, 1,    0.1],
                                   [2, 4, 2,    0.1]], 
                            columns = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']))
    >>> p = patterns.PatternMiner(df)
    >>> df_patterns = p.find(pattern = '=', parameters = {"min_confidence": 0.5,
                                                          "min_support"   : 2})

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

        new_df_patterns = derive_patterns(**kwargs, struct = self.metapatterns, dataframe = self.df_data)

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
        '''
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
        '''
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

def formula_expression(pattern):
    '''
    '''
    # the column name
    column_P = pattern[0]
    column_Q = pattern[2]
    # the content of the column
    content_P = pattern[3]
    content_Q = pattern[5]

    if pattern[1] != '-->':
        formula_string = '"' + str(column_P[0]) + '"' + str(pattern[1]) + ' "' + str(column_Q[0]) + '"'
    else:
        # if condition
        condition = '' 
        for idx, cond in enumerate(column_P):
            if type(content_P[idx]) == str:
                r_string = '"' + str(content_P[idx]) + '"'
            else:
                r_string = str(content_P[idx])
            condition = condition + '({' + str(column_P[idx]) + '} = ' + r_string + ')'
            if cond != column_P[-1]:
                condition = condition + ' and '

        # expression is if condition holds
        expression = ''
        for idx, cond in enumerate(column_Q):
            if type(content_Q[idx]) == str:
                r_string = '"' + str(content_Q[idx]) + '"'
            else:
                r_string = str(content_Q[idx])
            expression = expression + '("' + str(column_Q[idx]) + '" = ' + r_string + ')'
            if cond != column_Q[-1]:
                expression = expression + ' and '

        formula_string = "IF " + "(" + condition + ") THEN " + expression

    return formula_string

def pandas_expression(pattern, encode, result_type):
    '''
    '''
    # the column name
    column_P = pattern[0]
    column_Q = pattern[2]
    # the content of the column
    content_P = pattern[3]
    content_Q = pattern[5]

    if pattern[1] != '-->':
        if pattern[1]=="=":
            if result_type == False:
                string_pattern = "!="
            else:
                string_pattern = "=="
        elif pattern[1]=="sum":
            if result_type == False:
                string_pattern = "!="
            else:
                string_pattern = "=="
        else:
            string_pattern = str(pattern[1])

        pandas_string = 'df[('
        pandas_string = pandas_string + 'df["' + str(column_P[0]) + '"]'
        for p_item in column_P[1:]:
            pandas_string = pandas_string + '+ df["' + str(p_item) + '"]'
        pandas_string = pandas_string + ")" + string_pattern + 'df["' + str(column_Q[0]) + '"]]'

    else:
        # if condition
        condition_P = ""
        for idx, cond in enumerate(column_P):
            if type(content_P[idx]) == str:
                r_string = '"' + str(content_P[idx]) + '"'
            else:
                r_string = str(content_P[idx])

            if column_P[idx] in encode.keys():
                condition_P = condition_P + '(data_patterns.'+ encode[column_P[idx]]+ '(df["' + str(column_P[idx]) + '"])==' + r_string + ")"
            else:
                condition_P = condition_P + '(df["' + str(column_P[idx]) + '"]==' + r_string + ")"

            if cond != column_P[-1]:
                condition_P = condition_P + ' & '

        condition_Q = ""
        for idx, cond in enumerate(column_Q):
            if type(content_Q[idx]) == str:
                r_string = '"' + str(content_Q[idx]) + '"'
            else:
                r_string = str(content_Q[idx])

            if column_Q[idx] in encode.keys():
                condition_Q = condition_Q + '(data_patterns.'+ encode[column_Q[idx]]+ '(df["' + str(column_Q[idx]) + '"])==' + r_string + ")"
            else:
                condition_Q = condition_Q + '(df["' + str(column_Q[idx]) + '"]==' + r_string + ")"

            if cond != column_Q[-1]:
                condition_Q = condition_Q + ' & '

        if result_type == False:
            pandas_string = "df[" + condition_P + " & ~(" + condition_Q + ")]"
        else:
            pandas_string = "df[" + condition_P + " & (" + condition_Q + ")]"

    return pandas_string

def to_dataframe(patterns = None):
    '''
    '''
    # unpack pattern_id and pattern and patterns_stats and exclude co and ex and set pattern status to unknown
    patterns = list(patterns)
    if len(patterns) > 0:
        data = [pattern_id + 
                pattern + 
                pattern_stats + 
                [INITIAL_PATTERN_STATUS] + 
                [encode] + 
                [pandas_expression(pattern, encode, True)] +
                [pandas_expression(pattern, encode, False)] for [pattern_id, pattern, pattern_stats, encode] in patterns]

        df = PatternDataFrame(data = data, columns = PATTERNS_COLUMNS)
    else:
        df = PatternDataFrame(columns = PATTERNS_COLUMNS)

    return df

def update_statistics(dataframe = None, 
                      df_patterns = None):
    ''' 
    '''
    df_new_patterns = pd.DataFrame()

    if (dataframe is not None) and (df_patterns is not None):

        # adding the levels of the index to the columns (so they can be used for finding rules)
        for level in range(len(dataframe.index.names)):
            dataframe[dataframe.index.names[level]] = dataframe.index.get_level_values(level = level)

        for idx in df_patterns.index:

            if df_patterns.loc[idx, RELATION_TYPE]!="almost =":

                pandas_co = df_patterns.loc[idx, PANDAS_CO].replace("data_patterns.", "")
                pandas_ex = df_patterns.loc[idx, PANDAS_EX].replace("data_patterns.", "")
                n_co = len(eval(pandas_co, ENCODINGS_DICT, {'df': dataframe}).index.values.tolist())
                n_ex = len(eval(pandas_ex, ENCODINGS_DICT, {'df': dataframe}).index.values.tolist())
                total = n_co + n_ex
                if total > 0:
                    conf = n_co / total
                else:
                    conf = 0
                df_patterns.loc[idx, SUPPORT] = n_co
                df_patterns.loc[idx, EXCEPTIONS] = n_ex
                df_patterns.loc[idx, CONFIDENCE] = conf
                df_new_patterns = df_patterns

            else: # almost =
                pattern_id = [df_patterns.loc[idx, PATTERN_ID], df_patterns.loc[idx, CLUSTER]]
                pattern = [df_patterns.loc[idx, P_COLUMNS], 
                           df_patterns.loc[idx, RELATION_TYPE], 
                           df_patterns.loc[idx, Q_COLUMNS],
                           df_patterns.loc[idx, P_PART], 
                           df_patterns.loc[idx, RELATION], 
                           df_patterns.loc[idx, Q_PART]]

                new_patterns = list()
                preprocess_operator = preprocess[pattern[1]]
                c0 = dataframe.columns.get_loc(pattern[0][0]) # assuming only one element in lhs
                c1 = dataframe.columns.get_loc(pattern[2][0]) # assuming only one element in rhs
                nonzero = (dataframe.values != 0).T
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                data_array = dataframe.values[data_filter].T
                if data_array.size:
                    #confirmations of the pattern, a list of booleans
                    co = (np.abs(1 - data_array[c0,:] / data_array[c1,:]) <= 0.01)
                    pattern_data = derive_pattern_data(dataframe,
                                         [dataframe.columns[c0]], 
                                         [dataframe.columns[c1]], 
                                         pattern[1],
                                         pattern_id[1],
                                         co, 
                                         0, 
                                         data_filter)
                    new_patterns.append(pattern_data)
                df_new_patterns = to_dataframe(new_patterns)

        for level in range(len(dataframe.index.names)):
            del dataframe[dataframe.index.names[level]]

    return df_new_patterns

def derive_results(dataframe = None, 
                   P_dataframe = None,
                   Q_dataframe = None,
                   df_patterns = None):
    ''' 
    '''
    if (P_dataframe is not None) and (Q_dataframe is not None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            print("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []
    if (dataframe is not None) and (df_patterns is not None):
        # adding the levels of the index to the columns (so they can be used for finding rules)
        # for level in range(len(df.index.names)):
        #     df[df.index.names[level]] = df.index.get_level_values(level = level)
        # for pattern_id, pattern, pattern_stats, co, ex, encode in patterns:
        #     results.extend([[TEXT_EXCEPTION, pattern_id, item, pattern_stats, pattern, 
        #                   df.loc[item, pattern[0]].values.tolist(),
        #                   df.loc[item, pattern[2]].values.tolist()] for item in ex])
        #     results.extend([[TEXT_CONFIRMATION, pattern_id, item, pattern_stats, pattern,
        #                   df.loc[item, pattern[0]].values.tolist(),
        #                   df.loc[item, pattern[2]].values.tolist()] for item in co])


        df = dataframe.copy()

        results = []

        for idx in df_patterns.index:

            if df_patterns.loc[idx, RELATION_TYPE] != "almost =":

                pandas_ex = df_patterns.loc[idx, "pandas ex"].replace("data_patterns.", "")
                pandas_co = df_patterns.loc[idx, "pandas co"].replace("data_patterns.", "")
                results_ex = eval(pandas_ex, ENCODINGS_DICT, {'df': df}).index.values.tolist()
                results_co = eval(pandas_co, ENCODINGS_DICT, {'df': df}).index.values.tolist()
                for i in results_ex:
                    results.append([False, 
                                                    df_patterns.loc[idx, "pattern_id"], 
                                                    df_patterns.loc[idx, "cluster"], 
                                                    str(i), 
                                                    df_patterns.loc[idx, "support"], 
                                                    df_patterns.loc[idx, "exceptions"], 
                                                    df_patterns.loc[idx, "confidence"], 
                                                    df_patterns.loc[idx, "P columns"], 
                                                    df_patterns.loc[idx, "relation type"], 
                                                    df_patterns.loc[idx, "Q columns"], 
                                                    df_patterns.loc[idx, "P"], 
                                                    df_patterns.loc[idx, "relation"], 
                                                    df_patterns.loc[idx, "Q"], 
                                                    df.loc[i, df_patterns.loc[idx, "P columns"]].values.tolist(), 
                                                    df.loc[i, df_patterns.loc[idx, "Q columns"]].values.tolist()])
                for i in results_co:
                    results.append([True, 
                                                    df_patterns.loc[idx, "pattern_id"], 
                                                    df_patterns.loc[idx, "cluster"], 
                                                    str(i), 
                                                    df_patterns.loc[idx, "support"], 
                                                    df_patterns.loc[idx, "exceptions"], 
                                                    df_patterns.loc[idx, "confidence"], 
                                                    df_patterns.loc[idx, "P columns"], 
                                                    df_patterns.loc[idx, "relation type"], 
                                                    df_patterns.loc[idx, "Q columns"], 
                                                    df_patterns.loc[idx, "P"], 
                                                    df_patterns.loc[idx, "relation"], 
                                                    df_patterns.loc[idx, "Q"], 
                                                    df.loc[i, df_patterns.loc[idx, "P columns"]].values.tolist(), 
                                                    df.loc[i, df_patterns.loc[idx, "Q columns"]].values.tolist()])
        df_results = pd.DataFrame(data = results, columns = RESULTS_COLUMNS)
        df_results.sort_values(by = ["index", "confidence", "support"], ascending = [True, False, False], inplace = True)
        df_results.set_index(["index"], inplace = True)
        try:
            df_results.index = pd.MultiIndex.from_tuples(df_results.index)
        except:
            df_results.index = df_results.index

    return ResultDataFrame(df_results)

def derive_patterns(dataframe = None, 
                    struct    = None):
    ''' 
    '''
    for metapattern in struct:
        pattern = metapattern.get("pattern", "-->")
        if pattern == "-->":
            df_results = derive_association_patterns(metapattern = metapattern, dataframe = dataframe)
        else:
            df_results = derive_quantitative_patterns(dataframe = dataframe,
                                                      P_dataframe = metapattern.get("P_dataframe", None),
                                                      Q_dataframe = metapattern.get("Q_dataframe", None),
                                                      pattern = pattern,
                                                      pattern_name = metapattern.get("name", None),
                                                      columns = metapattern.get("columns", None),
                                                      P_columns = metapattern.get("P_columns", None),
                                                      Q_columns = metapattern.get("Q_columns", None),
                                                      value = metapattern.get("values", None),
                                                      parameters = metapattern.get("parameters", None))
    return df_results

def derive_association_patterns(metapattern = None,
                                dataframe = None):

    metapattern = copy.deepcopy(metapattern)

    new_list = list()

    p_part = metapattern.get("P_columns", None)
    q_part = metapattern.get("Q_columns", None)

    # there are four cases: 
    # - p and q are given, 
    # - p is given but q is not given, 
    # - q is given but p is not, 
    # - p and q are not given
    if ((p_part is None) and (q_part is not None)):
        p_set = [col for col in dataframe.columns if col not in metapattern["Q_columns"]]
        p_set = list(itertools.chain.from_iterable(itertools.combinations(p_set, n+1) for n in range(len(p_set))))
        for item in p_set:
            metapattern["P_columns"] = list(item)
            new_patterns = derive_patterns_from_metapattern(struct = metapattern, dataframe = dataframe)
            new_list.extend(new_patterns)
    elif ((q_part is None) and (p_part is not None)):
        q_set = [col for col in dataframe.columns if col not in metapattern["P_columns"]]
        q_set = list(itertools.chain.from_iterable(itertools.combinations(q_set, n+1) for n in range(len(q_set))))
        for item in q_set:
            metapattern[Q_PART] = list(item)
            new_patterns = derive_patterns_from_metapattern(struct = metapattern, dataframe = dataframe)
            new_list.extend(new_patterns)
    elif ((q_part is None) and (p_part is None)):
        p_set = [col for col in dataframe.columns]
        p_set = list(itertools.chain.from_iterable(itertools.combinations(p_set, n+1) for n in range(len(p_set))))            
        for p_item in p_set:
            q_set = [col for col in dataframe.columns if col not in p_item]
            q_set = list(itertools.chain.from_iterable(itertools.combinations(q_set, n+1) for n in range(len(q_set))))
            for q_item in q_set:
                metapattern["Q_columns"] = list(q_item)
                metapattern["P_columns"] = list(p_item)
                new_patterns = derive_patterns_from_metapattern(struct = metapattern, dataframe = dataframe)
                new_list.extend(new_patterns)
    else:
        new_patterns = derive_patterns_from_metapattern(struct = metapattern, dataframe = dataframe)
        new_list.extend(new_patterns)

    df_results = to_dataframe(patterns = new_list)

    return df_results

def derive_patterns_from_metapattern(dataframe = None, 
                                     struct    = None):
    ''' 
    '''
    # the struct contains the structure of the pattern as a dictionary
    P = sorted(struct["P_columns"])
    Q = sorted(struct["Q_columns"])
    df_features = dataframe.copy()
    # adding index levels to columns (in case the pattern contains index elements)
    for level in range(len(df_features.index.names)):
        df_features[df_features.index.names[level]] = df_features.index.get_level_values(level = level)
    # derive feature list from P and Q
    df_features = df_features[P + Q]
    # apply encoding of features
    if ENCODE in struct.keys():
        for c in struct[ENCODE].keys():
            df_features[c] = struct[ENCODE][c](df_features[c])
    df_patterns = df_features.reset_index(drop = True).drop_duplicates() # these are all unique combinations, i.e. the potential rules
    patterns = list()
    for row in df_patterns.index:
        df_selection = df_features
        for column in P:
            df_selection = df_selection[df_selection[column] == df_patterns.loc[row, column]]
        total = len(df_selection)
        if total > 0:
            df_co = df_selection
            for column in Q:
                df_co = df_co[df_co[column] == df_patterns.loc[row, column]]
            n_co = len(df_co.index)
            exceptions = [i for i in df_selection.index if not i in df_co.index]
            if len(exceptions) > 0:
                df_ex = df_selection.loc[exceptions]
                n_ex = len(df_ex.index)
            else:
                df_ex = pd.DataFrame()
                n_ex = 0
            conf = n_co / total
            if conf >= struct.get('threshold', 0.75):
                encode_str = '{}'
                if ENCODE in struct.keys():
                    encode_str = str(struct[ENCODE])
                    for string in re.findall("\<(.*?)\>", encode_str):
                        encode_str = encode_str.replace("<"+string+">", "'"+ re.findall("\s(.*?)\s", string)[0] + "'")
                patterns.append([[struct.get('name', "No name"), 0], 
                               [list(P), "-->", list(Q), list(df_patterns.loc[row, P].values), "-->", list(df_patterns.loc[row, Q].values)],
                                [n_co, n_ex, conf], ast.literal_eval(encode_str)])
    return patterns

def convert_columns(patterns = [], 
                    dataframe_input = None, 
                    dataframe_output = None):
    ''' 
    '''
    new_patterns = list()
    for pattern_id, pattern, pattern_stats, encode in patterns:
        new_pattern = [ [dataframe_output.columns[dataframe_input.columns.get_loc(item)] for item in pattern[0]],
                        pattern[1],
                        [dataframe_output.columns[dataframe_input.columns.get_loc(item)] for item in pattern[2]]] + pattern[3:6]
        new_encode = {dataframe_output.columns[dataframe_input.columns.get_loc(item)]: encode[item] for item in encode.keys()}
        new_patterns.append([pattern_id, new_pattern, pattern_stats, new_encode])
    return new_patterns

# equivalence -> reported together
def logical_equivalence(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ((nonzero_c1 & nonzero_c2) | (~nonzero_c1 & ~nonzero_c2))

# implication
def logical_implication(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return ~(nonzero_c1 & ~nonzero_c2)

def logical_or(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return (nonzero_c1 | nonzero_c2)

def logical_and(*c):
    nonzero_c1 = (c[0] != 0)
    nonzero_c2 = (c[1] != 0)
    return (nonzero_c1 & nonzero_c2)

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
              'almost =': operator.and_,
              '<->': operator.or_, 
              '-->': operator.or_}

def derive_pattern_statistics(co):
    # co_sum is the support of the pattern
    co_sum = co.sum()
    ex_sum = (~co).sum()
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
        data.extend(['{}']) # dict of encodings (empty)
    return data

def get_parameters(parameters):
    confidence = parameters.get("min_confidence", 0.75)
    support = parameters.get("min_support", 1)
    return confidence, support

# generate patterns of the form [c1] operator value where c1 is in columns

def patterns_column_value(dataframe = None, 
                          pattern   = None,
                          pattern_name = "value",
                          columns   = None,
                          value     = None,
                          parameters= {}):
    ''' 
    '''
    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T
    for c in columns:
        # confirmations and exceptions of the pattern, a list of booleans
        co = reduce(operators[pattern], [data_array[c, :], 0])
        pattern_data = derive_pattern_data(dataframe,
                                           dataframe.columns[c],
                                           value,
                                           pattern,
                                           pattern_name,
                                           co, 
                                           confidence, 
                                           None)
        if pattern_data and len(co) >= support:
            yield pattern_data

def patterns_column_column(dataframe  = None,
                           pattern    = None,
                           pattern_name = "column",
                           P_columns  = None, 
                           Q_columns  = None, 
                           parameters = {}):
    ''' 
    '''
    confidence, support = get_parameters(parameters)
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

def patterns_column_column_almost(dataframe  = None,
                                  pattern    = None,
                                  pattern_name = "column",
                                  P_columns  = None, 
                                  Q_columns  = None, 
                                  parameters = {}):
    ''' 
    '''
    confidence, support = get_parameters(parameters)
    almost_threshold = parameters.get("almost_threshold", 0.01)
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
                        co = (np.abs(1 - data_array[c0,:] / data_array[c1,:]) <= almost_threshold)
#                        reduce(operators[pattern], data_array[[c0, c1], :])
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
                         pattern    = None,
                         pattern_name = "sum",
                         parameters = {}):
    ''' 
    '''
    confidence, support = get_parameters(parameters)
    sum_elements = parameters.get("sum_elements", 2)
    preprocess_operator = preprocess[pattern]
    data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = (dataframe.values != 0).T
    n = len(dataframe.columns)
    matrix = np.ones(shape = (n, n), dtype = bool)
#    for c in itertools.combinations(range(n), 2):
        #v = (data_array[c[1], :] <= data_array[c[0], :] + 1).all()
        #matrix[c[0], c[1]] = v
        #matrix[c[1], c[0]] = ~v
    #np.fill_diagonal(matrix, False)

    for elements in range(2, sum_elements + 1):
        for sum_col in range(n):
            lower_columns, = np.where(matrix[sum_col] == True)
            for sum_parts in itertools.combinations(lower_columns, elements):
                subset = sum_parts + (sum_col,)
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in subset])
                data_array = dataframe.values[data_filter].T
                
                if data_array.size:
                    # determine sum of columns in subset
                    data_array[sum_col, :] = -data_array[sum_col, :]
                    co = (abs(reduce(operator.add, data_array[subset, :])) < 1)
                    pattern_data = derive_pattern_data(dataframe, 
                                        [dataframe.columns[c] for c in sum_parts],
                                        [dataframe.columns[sum_col]],
                                        pattern,
                                        pattern_name,
                                        co, 
                                        confidence, 
                                        None)
                    if pattern_data and len(co) >= support:
                        yield pattern_data

def derive_quantitative_patterns(dataframe    = None,
                                 P_dataframe  = None,
                                 Q_dataframe  = None,
                                 pattern      = None,
                                 pattern_name = None,
                                 columns      = None,
                                 P_columns    = None, 
                                 Q_columns    = None,
                                 value        = None,
                                 parameters   = {}):
    ''' 
    '''
    # if P_dataframe and Q_dataframe are given then join the dataframes and select columns
    if (P_dataframe is not None) and (Q_dataframe is not None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            print("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []
        P_columns = P_dataframe.columns
        Q_columns = Q_dataframe.columns

    # select all columns with numerical values
    numerical_columns = [dataframe.columns[c] for c in range(len(dataframe.columns)) 
                            if ((dataframe.dtypes[c] == 'float64') or (dataframe.dtypes[c] == 'int64')) and (dataframe.iloc[:, c] != 0).any()]
    dataframe = dataframe[numerical_columns]

    if P_columns is not None:
        P_columns = [dataframe.columns.get_loc(c) for c in P_columns if c in numerical_columns]
    else:
        P_columns = range(len(dataframe.columns))

    if Q_columns is not None:
        Q_columns = [dataframe.columns.get_loc(c) for c in Q_columns if c in numerical_columns]
    else:
        Q_columns = range(len(dataframe.columns))

    if columns is not None: 
        columns = [dataframe.columns.get_loc(c) for c in columns if c in numerical_columns]
    else:
        columns = range(len(dataframe.columns))

    # if a value is given -> columns pattern value
    if value is not None:
        results = patterns_column_value(dataframe = dataframe,
                                        pattern = pattern,
                                        pattern_name = pattern_name,
                                        columns = columns,
                                        value = value,
                                        parameters = parameters)
    elif pattern == 'sum':
        results = patterns_sums_column(dataframe = dataframe,
                                       pattern = pattern,
                                       pattern_name = pattern_name,
                                       parameters = parameters) 
    elif pattern == 'almost =':
        results = patterns_column_column_almost(dataframe = dataframe,
                                                pattern = pattern, 
                                                pattern_name = pattern_name,
                                                P_columns = P_columns,
                                                Q_columns = Q_columns,
                                                parameters = parameters)
    # everything else -> c1 pattern c2
    else:
        results = patterns_column_column(dataframe = dataframe,
                                         pattern = pattern, 
                                         pattern_name = pattern_name,
                                         P_columns = P_columns,
                                         Q_columns = Q_columns,
                                         parameters = parameters)
    return to_dataframe(patterns = results)

def read_excel(filename = None, 
               dataframe = None,
               sheet_name = 'Patterns'):
    df_patterns = pd.read_excel(filename, sheet_name = sheet_name)
    df_patterns.fillna('', inplace = True)
    df_patterns[RELATION_TYPE] = df_patterns[RELATION_TYPE].str[1:]
    patterns = list()
    for row in df_patterns.index:
        P_columns = evaluate_excel_string(df_patterns.loc[row, P_COLUMNS])
        Q_columns = evaluate_excel_string(df_patterns.loc[row, Q_COLUMNS])
        P = evaluate_excel_string(df_patterns.loc[row, P_PART])
        Q = evaluate_excel_string(df_patterns.loc[row, Q_PART])
        encode = ast.literal_eval(df_patterns.loc[row, ENCODINGS])
        if dataframe is not None:
            df_selection = dataframe
            for i, column in enumerate(P_columns):
                if column in encode.keys():
                    df_selection = df_selection[[elem == P[i] for elem in getattr(encodings, encode[column])(df_selection[column])]]
                else:
                    df_selection = df_selection[df_selection[column] == P[i]]
            total = len(df_selection)
            if total > 0:
                df_co = df_selection
                df_ex = df_selection
                for i, column in enumerate(Q_columns):
                    if column in encode.keys():
                        df_co = df_co[[elem == Q[i] for elem in getattr(encodings, encode[column])(df_co[column])]]
                    else:
                        df_co = df_co[df_co[column] == Q[i]]
                l = [i for i in df_ex.index if not i in df_co.index]
                if len(l) > 0:
                    df_ex = df_ex.loc[l]
                else:
                    df_ex = pd.DataFrame()
                n_co = len(df_co.index)
                n_ex = len(df_ex.index)
                conf = n_co / total
        else:
            n_co = 0
            n_ex = 0
            conf = 0
            df_co = pd.DataFrame()
            df_ex = pd.DataFrame()
        patterns.append([[df_patterns.loc[row, PATTERN_ID], 0], 
                         [P_columns, df_patterns.loc[row, RELATION_TYPE], Q_columns, 
                         P, df_patterns.loc[row, RELATION], Q],
                         [n_co, n_ex, conf], encode])
    return to_dataframe(patterns = patterns)

def evaluate_excel_string(s):
    if s != '':
        return ast.literal_eval(s)
    else:
        return s
