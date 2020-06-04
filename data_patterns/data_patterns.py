# -*- coding: utf-8 -*-

"""Main module."""

# imports
import pandas as pd
import numpy as np
import copy
import xlsxwriter
import ast
from functools import reduce
import itertools
import logging
from .constants import *
from .transform import *
from .encodings import *
from .parser import *

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
       In three flavours:
       - expressions (defined as a string),
       - conditional rules ('-->'-pattern defined with their columns) and
       - single rules (defined with their colums)
    '''
    df_patterns = pd.DataFrame(columns = PATTERNS_COLUMNS)
    for metapattern in metapatterns:
        if "expression" in metapattern.keys():
            patterns = derive_patterns_from_template_expression(metapattern = metapattern,
                                                                dataframe = dataframe)
        else:
            patterns = derive_patterns_from_code(metapattern = metapattern,
                                              dataframe = dataframe)
        df_patterns = df_patterns.append(patterns, ignore_index = True)

    df_patterns[CLUSTER] = df_patterns[CLUSTER].astype(np.int64)

    df_patterns[SUPPORT] = df_patterns[SUPPORT].astype(np.int64)
    df_patterns[EXCEPTIONS] = df_patterns[EXCEPTIONS].astype(np.int64)
    df_patterns.index.name = 'index'
    return PatternDataFrame(df_patterns)

def derive_patterns_from_template_expression(metapattern = None,
                                             dataframe = None):
    """
    Here we can add the constructions of expressions from an expression with wildcards
    """
    expression = metapattern.get("expression", "")
    parameters = metapattern.get("parameters", {})

    new_list = derive_patterns_from_expression(expression, metapattern, dataframe)
    df_patterns = to_dataframe(patterns = new_list, parameters = parameters)
    return df_patterns



def get_possible_columns(amount, expression, dataframe):
    if amount == 0:
        return [expression]

    all_columns = []

    for datapoint in re.findall(r'{.*?}', expression): # See which columns we are looking for per left open column
        d = datapoint[1:-1] # strip {" and "}
        if len(d) < 5:
            all_columns.append([re.search(d, col).group(0) for col in dataframe.columns if re.search(d, col)])
        else: # check for multiple options
            d = d[2:-2]
            d = d.strip().split(',')
            columns = []
            for item in d:
                item = item + '.*' # needed for regex
                for col in dataframe.columns:
                    if re.search(item, col):
                        columns.append(re.search(item, col).group(0))
            all_columns.append(columns)
        expression = expression.replace(datapoint, '{.*}', 1) # Replace it so that it goes well later
    if amount > 1: # Combine the lists into combinations where we do not have duplicates
        if re.search('AND', expression):
            possibilities = [p for p in itertools.product(*all_columns) if len(set(p)) == int(len(p)/2)]
        else:
            possibilities = [p for p in itertools.product(*all_columns) if len(set(p)) == len(p)]

        # We want to get rid of duplicates results when doing sum pattern or comparing columns, i.e.{x}={y} is the same as {y}={x}
        item = re.search(r'(.*)(=)(.*)', expression)
        if item:
            # only do this when the righthandside is a column
            if re.search(r'{(.*?)}',item.group(3)):
                d = {}
                for t in possibilities:
                    # Flatten the tuple
                    flat = itertools.chain.from_iterable(part if isinstance(part,list) else [part]
                                               for part in t)
                    maps_to = frozenset(flat) # Sets cannot be used as keys
                    d[maps_to] = t # Add it to the dict; the most recent addition "survives"

                possibilities = list(d.values())

    elif amount == 1: # If we have one empty spot, then just use the possible values
        possibilities = [[i] for i in all_columns[0]]

    possible_expressions = [] # list of all possible expressions
    for columns in possibilities:
        possible_expression = expression
        for column in columns: # replace with the possible column value
            possible_expression = possible_expression.replace(".*", '"' + column + '"', 1) # replace with column
        possible_expressions.append(possible_expression)
    return possible_expressions

def get_possible_values(amount, possible_expressions, dataframe):
    if amount < 1: # no values to be found
        return possible_expressions
    else:
        expressions = []
        for possible_expression in possible_expressions:
            all_columns = []
            for item in re.findall(r'.*?@', possible_expression): # See which columns we are looking for per left open column
                value_col = re.findall("{.*?}", item)[-1] # Get the column that matches the value indicator *@
                value_col = value_col[2:-2] # strip { and }
                all_columns.append(value_col)

            # print(all_columns)
            all_columns_v = dataframe[all_columns].drop_duplicates().to_numpy()
            # print(all_columns_v)
            for columns_v in all_columns_v: # Make all combinations without duplicates  of values
                possible_expression_v = possible_expression
                for column_v in columns_v:
                    if isinstance(column_v, str):
                        possible_expression_v = possible_expression_v.replace('"@"', '"'+ column_v +'"', 1) # replace adn add ""
                    else:
                        possible_expression_v = possible_expression_v.replace('"@"', str(column_v), 1) # replace with str
                expressions.append(possible_expression_v)
        return expressions

def add_qoutation(possible_expressions):
    new_expressions = []
    datapoints = []
    for expression in possible_expressions:
        for datapoint in re.findall(r'{.*?}', expression):
            if datapoint[1] != '"' and datapoint not in datapoints:
                d = datapoint[1:-1] # strip {" and "}
                expression = expression.replace(d, '"' + d +'"') # Replace it so that it goes well later
                datapoints.append(datapoint)
        new_expressions.append(expression)
    return new_expressions


def derive_patterns_from_expression(expression = "",
                                    metapattern = None,
                                    dataframe = None):
    """
    """
    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")
    encode = metapattern.get(ENCODE, {}) # TO DO
    encodings = get_encodings()
    confidence, support = get_parameters(parameters)
    solvency = parameters.get('solvency', False)
    patterns = list()

    amount = expression.count('.*}') #Amount of columns to be found
    amount_v = expression.count("@") #Amount of column values to be found

    possible_expressions = get_possible_columns(amount, expression, dataframe)
    possible_expressions = add_qoutation(possible_expressions)
    possible_expressions = get_possible_values(amount_v, possible_expressions, dataframe)
    for possible_expression in possible_expressions:
        # print(possible_expression)
        pandas_expressions = to_pandas_expressions(possible_expression, encode, parameters, dataframe)
        print(pandas_expressions)
        try: # Some give error so we use try
            n_co = len(eval(pandas_expressions[0], encodings, {'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
            n_ex = len(eval(pandas_expressions[1], encodings, {'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
            conf = np.round(n_co / (n_co + n_ex + 1e-11), 4)
            if ((conf >= confidence) and (n_co >= support)):
                xbrl_expressions = to_xbrl_expressions(possible_expression, encode, parameters)
                patterns.extend([[[name, 0], possible_expression, [n_co, n_ex, conf]] + pandas_expressions + xbrl_expressions + ['']])
        except TypeError as e:
            if solvency:
                patterns.extend([[[name, 0], possible_expression, [0, 0, 0]] + ['', ''] + ['', ''] + [str(e)]])
            else:
                continue
        except:
            if solvency:
                patterns.extend([[[name, 0], possible_expression, [0, 0, 0]] + ['',''] + ['', ''] + ['UNKNOWN ERROR']])
            else:
                continue

    return patterns

def derive_patterns_from_code(metapattern = None,
                                dataframe = None):
    '''Derive conditional rule patterns
       If no columns are given, then the algorithm searches for all possibilities
    '''
    patterns = list()


    P_dataframe = metapattern.get("P_dataframe", None)
    Q_dataframe = metapattern.get("Q_dataframe", None)
    pattern = metapattern.get("pattern", None)
    pattern_name = metapattern.get("name", None)
    columns = metapattern.get("columns", None)
    P_columns = metapattern.get("P_columns", None)
    Q_columns = metapattern.get("Q_columns", None)
    value = metapattern.get("value", None)
    values = metapattern.get("values", None)
    parameters = metapattern.get("parameters", {})



    if pattern == '-->':
        possible_expressions = derive_conditional_pattern(metapattern = metapattern, dataframe = dataframe)

    # everything else -> c1 pattern c2
    else:
        possible_expressions = derive_quantitative_pattern(
        metapattern = metapattern, dataframe = dataframe,
                                        pattern = pattern,
                                        pattern_name = pattern_name,
                                         P_columns = P_columns,
                                         Q_columns = Q_columns,
                                        columns = columns,
                                        value = value,
                                        parameters = parameters)


    for expression in possible_expressions:
        patterns.extend(derive_patterns_from_expression(expression, metapattern, dataframe))

    df_patterns = to_dataframe(patterns = patterns, parameters = parameters)
    return df_patterns


def derive_conditional_pattern(dataframe = None,
                               metapattern = None):
    '''Here we derive the patterns from the metapattern definitions
       by evaluating the pandas expressions of all potential patterns
    '''
    # get items from metapattern definition
    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")
    encode = metapattern.get(ENCODE, {})
    P_columns = metapattern.get("P_columns", list(dataframe.columns.values))
    Q_columns = metapattern.get("Q_columns", list(dataframe.columns.values))
    P_values = metapattern.get("P_values", ['@']*len(P_columns)) # in case we do not have values
    Q_values = metapattern.get("Q_values", ['@']*len(Q_columns))

    confidence, support = get_parameters(parameters)

    # derive df_feature list from P and Q (we use a copy, so we can change values for encodings)
    df_features = dataframe[P_columns + Q_columns].copy()
    # execute dynamic encoding functions
    encodings = get_encodings()
    # perform encodings on df_features
    if encode != {}:
        for c in df_features.columns:
            if c in encode.keys():
                df_features[c] = eval(str(encode[c])+ "(s)", encodings, {'s': df_features[c]})
    if encode != {}: # only use when P value is not given
        expressions = []
        df_features = df_features.drop_duplicates(P_columns + Q_columns)
        for idx in range(len(df_features.index)):
            P_values = list(df_features[P_columns].values[idx])
            Q_values = list(df_features[Q_columns].values[idx])
            expression = generate_conditional_expression(P_columns, P_values, Q_columns, Q_values, parameters)

            expressions.append(expression)

        return expressions
    # In the case that P and Q values are both given, we only want to compute it for these values and not search for other values like above
    expression = generate_conditional_expression(P_columns, P_values, Q_columns, Q_values, parameters)
    return [expression]


def get_parameters(parameters):
    confidence = parameters.get("min_confidence", 0.75)
    support = parameters.get("min_support", 2)
    return confidence, support

def derive_quantitative_pattern(metapattern = None,
                                dataframe = None,
                                pattern = None,
                                pattern_name = "quantitative",
                                 P_columns = None,
                                 Q_columns = None,
                                columns = None,
                                value = None,
                                parameters = {}):

    confidence, support = get_parameters(parameters)
    decimal = parameters.get("decimal", 8)
    P_dataframe = metapattern.get("P_dataframe", None)
    Q_dataframe = metapattern.get("Q_dataframe", None)

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
                            if ((dataframe.dtypes[c] == 'float64') or (dataframe.dtypes[c] == 'int64'))]
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
        
    data_array = dataframe.values.T
    patterns = list()
    if value is not None:
        for c in columns:
            # confirmations and exceptions of the pattern, a list of booleans
            try:
                pattern_def = generate_single_expression([dataframe.columns[c]], value, pattern)
                patterns.append(pattern_def)
            except:
                continue

    elif pattern == 'percentile':
        percentile = parameters['percentile']
        add_per = (100-percentile)/2
        for c in columns:
            # confirmations and exceptions of the pattern, a list of booleans
            try:
                upper = round(np.percentile(data_array[c, :], percentile + add_per),2)
                lower = round(np.percentile(data_array[c, :], add_per),2)

                # confirmations and exceptions of the pattern, a list of booleans
                pattern_def = generate_single_expression([dataframe.columns[c]], [lower, upper], pattern)
                patterns.append(pattern_def)
            except:
                continue

    elif pattern == 'sum':
        sums = patterns_sums_column(dataframe  = dataframe,
                                 pattern_name = pattern_name,
                                 P_columns  = P_columns,
                                 Q_columns  = Q_columns,
                                 parameters = parameters)
        patterns = [pat for pat in sums]
    elif pattern == 'ratio':
        # TO DO
        return
    else:
        compares = patterns_column_column(dataframe  = dataframe,
                                pattern = pattern,
                                 pattern_name = pattern_name,
                                 P_columns  = P_columns,
                                 Q_columns  = Q_columns,
                                 parameters = parameters)
        patterns = [pat for pat in compares]
    return patterns

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
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = initial_data_array != 0
    operators = {'>' : operator.gt,
             '<' : operator.lt,
             '>=': operator.ge,
             '<=': operator.le,
             '=' : operator.eq,
             '!=': operator.ne,
             '<->': logical_equivalence,
             '-->': logical_implication}
    preprocess_operator = preprocess[pattern]

    for c0 in P_columns:
        for c1 in Q_columns:
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                if data_filter.any():
                    data_array = initial_data_array[:, data_filter]
                    if data_array.any():
                        pattern_def = generate_single_expression([dataframe.columns[c0]], [dataframe.columns[c1]], pattern)
                        yield pattern_def

def patterns_sums_column( dataframe  = None,
                         pattern_name = None,
                         P_columns  = None,
                         Q_columns  = None,
                         parameters = {}):
    '''Generate patterns of the form sum [c1-list] = [c2] where c1-list is column list and c2 is column
    '''
    confidence, support = get_parameters(parameters)
    sum_elements = parameters.get("sum_elements", 2)
    decimal = parameters.get("decimal", 0)
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
            try:
                start_array[rhs_column, :] = -start_array[rhs_column, :]
            except:
                continue
            lhs_column_list = [col for col in P_columns if (col != rhs_column)]
            for lhs_columns in itertools.combinations(lhs_column_list, lhs_elements):
                all_columns = lhs_columns + (rhs_column,)
                data_filter = np.logical_and.reduce(nonzero[all_columns, :])
                if data_filter.any():
                    pattern_def = generate_single_expression([dataframe.columns[c] for c in lhs_columns], [dataframe.columns[rhs_column]], 'sum')
                    yield pattern_def

def derive_ratio_pattern(dataframe  = None,
                   pattern_name = None,
                   P_columns  = None,
                   Q_columns  = None,
                   parameters = {}):
    """Generate patterns with ratios TODO: Needs big change!!
    """
    confidence, support = get_parameters(parameters)
    limit_denominator = parameters.get("limit_denominator", 10000000)
    decimal = parameters.get("decimal", 8)
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


def to_pandas_expressions(pattern, encode, parameters, dataframe):
    """Derive pandas code from the pattern definition string both confirmation and exceptions"""

    # preprocessing step
    res = preprocess_pattern(pattern, parameters)
    # datapoints to pandas, i.e. {column} -> df[column]
    res, nonzero_col = datapoints2pandas(res, encode)
    # expression to pandas, i.e. IF X=x THEN Y=y -> df[df[X]=x & df[Y]=y] for confirmations
    co_str, ex_str = expression2pandas(res, nonzero_col, parameters)

    return [co_str, ex_str]

def to_dataframe(patterns = None, parameters = {}):
    '''Convert list of patterns to dataframe with patterns
    '''
    # unpack pattern_id and pattern and patterns_stats and exclude co and ex and set pattern status to unknown
    patterns = list(patterns)
    if len(patterns) > 0:
        data = [pattern_id + [pattern] + pattern_stats + [INITIAL_PATTERN_STATUS] + [{}] +
               [pandas_co, pandas_ex, xbrl_co, xbrl_ex, error] for [pattern_id, pattern, pattern_stats, pandas_co, pandas_ex, xbrl_co, xbrl_ex, error] in patterns]
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
            n_co = len(eval(pandas_co, encodings,{'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
            n_ex = len(eval(pandas_ex, encodings, {'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
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
            pandas_ex = df_patterns.loc[idx, PANDAS_EX]
            pandas_co = df_patterns.loc[idx, PANDAS_CO]
            results_ex = eval(pandas_ex, encodings, {'df': df}).index.values.tolist()
            results_co = eval(pandas_co, encodings, {'df': df}).index.values.tolist()
            for i in results_ex:
                # values_p = df.loc[i, df_patterns.loc[idx, P_COLUMNS]].values.tolist()
                # if type(df_patterns.loc[idx, Q_COLUMNS])==list:
                #     values_q = df.loc[i, df_patterns.loc[idx, Q_COLUMNS]].values.tolist()
                # else:
                #     values_q = df_patterns.loc[idx, Q_COLUMNS]
                values_p = ""
                values_q = ""
                results.append([False,
                                df_patterns.loc[idx, "pattern_id"],
                                df_patterns.loc[idx, "cluster"],
                                i,
                                df_patterns.loc[idx, "support"],
                                df_patterns.loc[idx, "exceptions"],
                                df_patterns.loc[idx, "confidence"],
                                df_patterns.loc[idx, "pattern_def"],
                                values_p,
                                values_q])
            for i in results_co:
                # values_p = df.loc[i, df_patterns.loc[idx, P_COLUMNS]].values.tolist()
                # if type(df_patterns.loc[idx, Q_COLUMNS])==list:
                #     values_q = df.loc[i, df_patterns.loc[idx, Q_COLUMNS]].values.tolist()
                # else:
                #     values_q = df_patterns.loc[idx, Q_COLUMNS]
                values_p = ""
                values_q = ""
                results.append([True,
                                df_patterns.loc[idx, "pattern_id"],
                                df_patterns.loc[idx, "cluster"],
                                i,
                                df_patterns.loc[idx, "support"],
                                df_patterns.loc[idx, "exceptions"],
                                df_patterns.loc[idx, "confidence"],
                                df_patterns.loc[idx, "pattern_def"],
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


def read_excel(filename = None,
               dataframe = None,
               sheet_name = 'Patterns'):
    df = pd.read_excel(filename, sheet_name = sheet_name)
    df.fillna('', inplace = True)
    # df[RELATION_TYPE] = df[RELATION_TYPE].str[1:]
    patterns = list()
    for row in df.index:
        print(df.loc[row, PATTERN_DEF])
        pattern_def = df.loc[row, PATTERN_DEF]
        encode = ast.literal_eval(df.loc[row, ENCODINGS])
        pandas_co = df.loc[row, PANDAS_CO]
        pandas_ex = df.loc[row, PANDAS_EX]
        xbrl_co = df.loc[row, XBRL_CO]
        xbrl_ex = df.loc[row, XBRL_EX]
        patterns.append([[df.loc[row, PATTERN_ID], 0],
                         pattern_def,
                         [0, 0, 0], pandas_co, pandas_ex, xbrl_co, xbrl_ex])
    df_patterns = to_dataframe(patterns = patterns, parameters = {})
    if dataframe is not None:
        df_patterns = update_statistics(dataframe = dataframe, df_patterns = df_patterns)
    return df_patterns

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
