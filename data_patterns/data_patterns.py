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
import logging
logging.basicConfig(filename='logger.log',format='%(levelname)s:%(message)s',level=logging.DEBUG)
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
        self.df_results = None
        self.__process_parameters(*args, **kwargs)

    def find(self, *args, **kwargs):
        '''General function to find patterns
        '''
        logger = logging.getLogger(__name__)

        self.__process_parameters(*args, **kwargs)
        assert self.metapatterns is not None, "No patterns defined."
        assert self.df_data is not None, "No dataframe defined."
        logger.info('Rows in data: ' + str(self.df_data.shape[0]))

        new_df_patterns = derive_patterns(**kwargs, metapatterns = self.metapatterns, dataframe = self.df_data)

        if (not kwargs.get('append', False)) or (self.df_patterns is None):
            self.df_patterns = new_df_patterns
        else:
            if len(new_df_patterns.index) > 0:
                self.df_patterns.append(new_df_patterns)

        return self.df_patterns


    def correct_data(self, *args, **kwargs):
        '''General function to change data to correct value. This only works for a very simple conditional pattern. '''

        self.__process_parameters(*args, **kwargs)

        assert self.df_patterns is not None, "No patterns defined."
        assert self.df_data is not None, "No data defined."
        assert self.df_results is not None, "No analyzing data defined."


        df_data = self.df_data.copy()
        df_results = self.df_results.copy()
        df_results['correct_value'] = df_results['pattern_def'].apply(lambda x: get_value(x, num=2))
        df_results = df_results.loc[df_results['result_type'] == False]

        # Get the column names
        colq = get_value(df_results['pattern_def'].iloc[0], 2, 1)

        df_data.loc[df_data.index.isin(df_results.index), colq] = df_results['correct_value']
        return df_data, df_results # changed data and log what changed


    def analyze(self, *args, **kwargs):
        '''General function to analyze data given a list of patterns
        '''
        self.__process_parameters(*args, **kwargs)

        assert self.df_patterns is not None, "No patterns defined."
        assert self.df_data is not None, "No data defined."

        self.df_patterns = update_statistics(dataframe = self.df_data, df_patterns = self.df_patterns)

        self.df_results = derive_results(**kwargs, df_patterns = self.df_patterns, dataframe = self.df_data)

        return self.df_results

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

def get_value(pattern, num = 1, col=3):
    ''' Derive the P (num=1) or Q (num=2) value from a pattern'''

    values = []

    item = re.search(r'IF(.*)THEN(.*)', pattern)

    # If not conditional statement, return None
    if item == None:
        item2 = re.search(r'(.*)[><=](.*)', pattern)

        if pattern.count('}') > 2: # sum
            for match in re.finditer(r'{(.*?)}', item2.group(num)):
                item3 = re.search(r'"(.*)"', match.group(1))
                values.append(item3.group(1))

        else:
            if '{' in item2.group(2):
                item3 = re.search(r'{(.*?)}', item2.group(num))
                values.append(item3.group(1)[1:-1])

            else:
                if num == 1:
                    item3 = re.search(r'{(.*?)}', item2.group(1))
                    values.append(item3.group(1)[1:-1])
                else:
                    return None
    else:
        if col == 1:
            for match in re.finditer(r'{(.*?)}', item.group(num)):
                item3 = match.group(1)[1:-1]
                values.append(item3)
        else:
        # Find repeating pattern of conditions
            for match in re.finditer(r'(.*?)[&|\||\^](\s*\({.*)', item.group(num)):
                item2 = re.search(r'(.*)([>|<|!=|<=|>=|=])(.*)', match.group(1))
                item3 = re.search(r'"(.*)"', item2.group(col))

                if item3 is not None: # If string
                    values.append(item3.group(1))
                else: # If float
                    values.append(float(item2.group(col).replace(')', '')))


            item2 = re.search(r'(.*)([>|<|!=|<=|>=|=])(.*)', item.group(num))
            item3 = re.search(r'"(.*)"', item2.group(col))

            if item3 is not None: # If string
                values.append(item3.group(1))
            else: # If float
                values.append(float(item2.group(col).replace(')', '')))

    if len(values) == 1:
        return values[0]
    else: # return list if multiple columns/values
        return values

def derive_patterns(dataframe   = None,
                    metapatterns = None):
    '''Derive patterns from metapatterns
       In three flavours:
       - expressions (defined as a string),
       - conditional rules ('-->'-pattern defined with their columns) and
       - single rules (defined with their colums)
    '''
    logger = logging.getLogger(__name__)
    logger.info('Find started ...')

    df_patterns = pd.DataFrame(columns = PATTERNS_COLUMNS)
    dataframe = dataframe.reset_index()
    for metapattern in metapatterns:
        parameters = metapattern.get("parameters", {})
        if "expression" in metapattern.keys():
            patterns = derive_patterns_from_template_expression(metapattern = metapattern,
                                                                dataframe = dataframe)
        else:
            patterns = derive_patterns_from_code(metapattern = metapattern,
                                              dataframe = dataframe)
        logger.info('Total rows in patterns: ' + str(patterns.shape[0]))
        if 'min_confidence' in parameters:
            if parameters['min_confidence'] == 'highest': # For when you want the highest confidence in a pattern
                patterns = get_highest_conf(patterns)
                logger.info('Reduction of rows in patterns to: ' + str(patterns.shape[0]))

        df_patterns = df_patterns.append(patterns, ignore_index = True)

    df_patterns[CLUSTER] = df_patterns[CLUSTER].astype(np.int64)
    df_patterns[SUPPORT] = df_patterns[SUPPORT].astype(np.int64)
    df_patterns[EXCEPTIONS] = df_patterns[EXCEPTIONS].astype(np.int64)
    df_patterns.index.name = 'index'

    logger.info('Find ended ...')

    return PatternDataFrame(df_patterns)

def get_highest_conf(df_patterns):

    df_patterns['P_val'] = df_patterns['pattern_def'].apply(lambda x: get_value(x, num=1))

    df_patterns = df_patterns.sort_values('support', ascending=False) # Sort values

    _, idx = np.unique(df_patterns['P_val'].astype(str).values, return_index=True) # Drop duplicate rows

    df_patterns = df_patterns.iloc[idx].sort_index() # Only get dataframe rows from these indices
    df_patterns = df_patterns.drop(['P_val'],1)
    return df_patterns

def derive_patterns_from_template_expression(metapattern = None,
                                             dataframe = None):
    """
    Here we can add the constructions of expressions from an expression with wildcards
    """
    expression = metapattern.get("expression", "")
    parameters = metapattern.get("parameters", {})
    encodings = metapattern.get("encode", {})
    solvency = parameters.get('solvency', False)
    if re.search(r'IF(.*)THEN(.*)', expression):
        new_list = derive_patterns_from_expression(expression, metapattern, dataframe)
    else:
        new_list = derive_quantitative_pattern_expression(expression, metapattern, dataframe)
    df_patterns = to_dataframe(patterns = new_list, parameters = parameters, encodings=encodings)
    return df_patterns


def derive_quantitative_pattern_expression(expression, metapattern, dataframe):
    logger = logging.getLogger(__name__)

    logger.info('Calling function derive_quantitative_pattern_expression')

    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")

    confidence, support = get_parameters(parameters)
    numerical_columns = [dataframe.columns[c] for c in range(len(dataframe.columns))
                            if ((dataframe.dtypes[c] == 'float64') or (dataframe.dtypes[c] == 'int64'))]
    dataframe = dataframe[numerical_columns]

    patterns = list()

    if '+' in expression: # Sum pattern
        sum_elements = expression.count('+') + 1
        parameters['sum_elements'] = sum_elements
        P_columns = []
        Q_columns = []
        for match in re.finditer(r'(.*?)[+|=]', expression): # Get P_columns
            possible_col = get_possible_columns(match.group(1).count('.*'), match.group(1),dataframe,True)
            P_columns = P_columns + possible_col
            P_columns = list(set(P_columns))
        for match in re.finditer(r'=(.*)', expression): # Get Q_columns
            possible_col = get_possible_columns(match.group(1).count('.*'), match.group(1),dataframe,True)
            Q_columns = Q_columns + possible_col
            Q_columns = list(set(Q_columns))

        # Right format
        Q_columns = [dataframe.columns.get_loc(c) for c in Q_columns if c in numerical_columns]
        P_columns = [dataframe.columns.get_loc(c) for c in P_columns if c in numerical_columns]
        Q_columns.sort()
        P_columns.sort()

        # Use numpy now
        sums = patterns_sums_column(dataframe  = dataframe,
                                 pattern_name = name,
                                 P_columns  = P_columns,
                                 Q_columns  = Q_columns,
                                 parameters = parameters)
        for pat in sums:
            patterns.extend(pat)
    else:
        item2 = re.search(r'(.*?)([!=|<=|>=|>|<|=])(.*)', expression)
        pattern = item2.group(2)
        item3 = item2.group(3)

        if item3[0] == '=': # Stupid bug that does not capture >= or <=, so fixed it like this
            pattern += '='
            item3 = item3[1:]

        if '{' in item3:
            P_columns = get_possible_columns(item2.group(1).count('.*'), item2.group(1),dataframe,True)
            Q_columns = get_possible_columns(item3.count('.*'), item3, dataframe,True)
            Q_columns = [dataframe.columns.get_loc(c) for c in Q_columns if c in numerical_columns]
            P_columns = [dataframe.columns.get_loc(c) for c in P_columns if c in numerical_columns]
            Q_columns.sort()
            P_columns.sort()
            compares = patterns_column_column(dataframe  = dataframe,
                                    pattern = pattern,
                                     pattern_name = name,
                                     P_columns  = P_columns,
                                     Q_columns  = Q_columns,
                                     parameters = parameters)
            for pat in compares:
                patterns.extend(pat)

        else:
            columns = get_possible_columns(item2.group(1).count('.*'), item2.group(1),dataframe,True)
            value = float(item3)
            columns = [dataframe.columns.get_loc(c) for c in columns if c in numerical_columns]
            columns.sort()
            values = patterns_column_value(dataframe  = dataframe,
                                     value = value,
                                     pattern = pattern,
                                     pattern_name = name,
                                     columns = columns,
                                     parameters = parameters)
            for val in values:
                patterns.extend(val)
    return patterns

def get_possible_columns(amount, expression, dataframe, quant=False):
    if amount == 0:
        if quant:
            return [expression.strip()[1:-1]]
        else:
            return [expression]

    all_columns = []

    for datapoint in re.findall(r'{.*?}', expression): # See which columns we are looking for per left open column
        d = datapoint[1:-1] # strip {" and "}
        if '*' not in datapoint:
            continue
        d = d.strip().split(',')
        columns = []
        for item in d:
            for col in dataframe.columns:
                if re.search(item, col):
                    columns.append(re.search(item, col).group(0))
        all_columns.append(columns)
        expression = expression.replace(datapoint, '{.*}', 1) # Replace it so that it goes well later

    if quant: # for quantitative expressions
        return all_columns[0]

    if amount > 1: # Combine the lists into combinations where we do not have duplicates
        if re.search('AND', expression):
            possibilities = [p for p in itertools.product(*all_columns) if len(set(p)) == int(len(p)/2)]
        else:
            possibilities = [p for p in itertools.product(*all_columns) if len(set(p)) == len(p)]


    elif amount == 1: # If we have one empty spot, then just use the possible values
        possibilities = [[i] for i in all_columns[0]]

    possible_expressions = [] # list of all possible expressions
    for columns in possibilities:
        possible_expression = expression
        for column in columns: # replace with the possible column value
            possible_expression = possible_expression.replace("{.*}", '{"' + column + '"}', 1) # replace with column
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

            all_columns_v = dataframe[all_columns].drop_duplicates().values

            items =  re.findall(r'\[(.*?@)\]', possible_expression) # Find the columns
            del_rows = []
            for i in range(len(items)):
                if len(items[i]) > 1: # only when we have a string
                    item = items[i][:-1]
                    for j in range(len(all_columns_v)): # check if each column can be used
                        if re.search(item, all_columns_v[j][i]) is not None:
                            if re.search(item, all_columns_v[j][i])[0] != all_columns_v[j][i]:
                                del_rows.append(j)
                        else:
                                del_rows.append(j)

                    possible_expression = possible_expression.replace(item, '', 1) # Replace it so that it goes well later

            all_columns_v = np.delete(all_columns_v, del_rows,0) # del rows


            for columns_v in all_columns_v:
                possible_expression_v = possible_expression
                for column_v in columns_v:
                    if isinstance(column_v, str):
                        possible_expression_v = possible_expression_v.replace('[@]', '"'+ column_v +'"', 1) # replace adn add ""
                    else:
                        possible_expression_v = possible_expression_v.replace('[@]', str(column_v), 1) # replace with str
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
    logger = logging.getLogger(__name__)

    logger.info('Calling function derive_patterns_from_expression')

    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")
    encode = metapattern.get(ENCODE, {})
    encodings = get_encodings()
    confidence, support = get_parameters(parameters)
    if confidence == 'highest':
        confidence = 0

    solvency = parameters.get('solvency', False)
    patterns = list()

    amount = expression.count('.*}') #Amount of columns to be found
    amount_v = expression.count("@") #Amount of column values to be found


    if metapattern.get('expression', None):
        df_features = dataframe.copy()
        # execute dynamic encoding functions
        if encode != {}:
            for c in df_features.columns:
                if c in encode.keys():
                    df_features[c] = eval(str(encode[c])+ "(s)", encodings, {'s': df_features[c]})
        dataframe2 = df_features

    else:
        dataframe2 = dataframe

    possible_expressions = get_possible_columns(amount, expression, dataframe2)
    possible_expressions = add_qoutation(possible_expressions)
    possible_expressions = get_possible_values(amount_v, possible_expressions, dataframe2)
    if len(possible_expressions) > 40:
        logger.warning(' Amount of possibilities is high! Namely, ' + str(len(possible_expressions)))
    else:
        logger.info(' Amount of possibilities: ' + str(len(possible_expressions)))

    for possible_expression in possible_expressions:
        # print(possible_expression)
        pandas_expressions = to_pandas_expressions(possible_expression, encode, parameters, dataframe)
        # print(pandas_expressions)
        try: # Some give error so we use try
            n_co = len(eval(pandas_expressions[0], encodings, {'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
            n_ex = len(eval(pandas_expressions[1], encodings, {'df': dataframe, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index)
            conf = np.round(n_co / (n_co + n_ex + 1e-11), 4)
            # print(n_co,n_ex)
            if ((conf >= confidence) and (n_co >= support)):
                xbrl_expressions = to_xbrl_expressions(possible_expression, encode, parameters)
                patterns.extend([[[name, 0], possible_expression, [n_co, n_ex, conf]] + pandas_expressions + xbrl_expressions + ['']])
                logger.info("Pattern " + name + " correctly parsed (#co=" + str(co) + ", #ex=" + str(ex) + ")")
        except TypeError as e:
            logger.error('Error in trying to process pandas expression: ' +  str(e))
            if solvency:
                patterns.extend([[[name, 0], possible_expression, [0, 0, 0]] + ['', ''] + ['', ''] + [str(e)]])
            else:
                continue
        except:
            logger.error('Error in trying to process pandas expression: UNKNOWN ERROR')
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
    encodings = metapattern.get("encode", {})
    parameters = metapattern.get("parameters", {})



    if pattern == '-->':
        possible_expressions = derive_conditional_pattern(metapattern = metapattern, dataframe = dataframe)
        for expression in possible_expressions:
            patterns.extend(derive_patterns_from_expression(expression, metapattern, dataframe))
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


        patterns = possible_expressions
    df_patterns = to_dataframe(patterns = patterns, parameters = parameters, encodings= encodings)
    return df_patterns


def derive_conditional_pattern(dataframe = None,
                               metapattern = None):
    '''Here we derive the patterns from the metapattern definitions
       by evaluating the pandas expressions of all potential patterns
    '''
    logger = logging.getLogger(__name__)

    logger.info(' Calling function derive_conditional_pattern')

    # get items from metapattern definition
    parameters = metapattern.get("parameters", {})
    name = metapattern.get('name', "No name")
    encode = metapattern.get(ENCODE, {})
    P_columns = metapattern.get("P_columns", list(dataframe.columns.values))
    Q_columns = metapattern.get("Q_columns", list(dataframe.columns.values))
    P_values = metapattern.get("P_values", ['[@]']*len(P_columns)) # in case we do not have values
    Q_values = metapattern.get("Q_values", ['[@]']*len(Q_columns))

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

        expressions = []
        df_features = df_features.drop_duplicates(P_columns + Q_columns)
        for idx in range(len(df_features.index)):
            P_values = list(df_features[P_columns].values[idx])
            Q_values = list(df_features[Q_columns].values[idx])
            expression = generate_conditional_expression(P_columns, P_values, Q_columns, Q_values, parameters)
            expression = expression.replace('"[@]"', '[@]')

            expressions.append(expression)

        return expressions
    # In the case that P and Q values are both given, we only want to compute it for these values and not search for other values like above
    expression = generate_conditional_expression(P_columns, P_values, Q_columns, Q_values, parameters)
    expression = expression.replace('"[@]"', '[@]')
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
    logger = logging.getLogger(__name__)

    logger.info(' Calling function derive_quantitative_pattern')

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
        values = patterns_column_value(dataframe  = dataframe,
                                value = value,
                                pattern = pattern,
                                 pattern_name = pattern_name,
                                 columns = columns,
                                 parameters = parameters)
        for val in values:
            patterns.extend(val)

    elif pattern == 'percentile':
        values = patterns_percentile(dataframe  = dataframe,
                                pattern = pattern,
                                 pattern_name = pattern_name,
                                 columns = columns,
                                 parameters = parameters)
        for val in values:
            patterns.extend(val)

    elif pattern == 'sum':
        sums = patterns_sums_column(dataframe  = dataframe,
                                 pattern_name = pattern_name,
                                 P_columns  = P_columns,
                                 Q_columns  = Q_columns,
                                 parameters = parameters)
        for pat in sums:
            patterns.extend(pat)

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
        for pat in compares:
            patterns.extend(pat)


    return patterns

operators = {'>' : operator.gt,
         '<' : operator.lt,
         '>=': operator.ge,
         '<=': operator.le,
         '=' : operator.eq,
         '!=': operator.ne,
         '<->': logical_equivalence,
         '-->': logical_implication}

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

def patterns_column_value(dataframe  = None,
                            value = None,
                           pattern    = None,
                           pattern_name = "value",
                           columns = None,
                           parameters = {}):

    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T

    for c in columns:
        co = reduce(operators[pattern], [data_array[c, :], value])
        co_sum, ex_sum, conf = derive_pattern_statistics(co)

        if (conf >= confidence) and (co_sum >= support):
            possible_expression = generate_single_expression([dataframe.columns[c]], value, pattern)
            pandas_expressions = to_pandas_expressions(possible_expression, {}, parameters, dataframe)
            xbrl_expressions = to_xbrl_expressions(possible_expression, {}, parameters)
            pattern_data = [[[pattern_name, 0], possible_expression, [co_sum, ex_sum, conf]] + pandas_expressions + xbrl_expressions + ['']]
            yield pattern_data

def patterns_percentile(dataframe  = None,
                           pattern    = None,
                           pattern_name = "percentile",
                           columns = None,
                           parameters = {}):

    confidence, support = get_parameters(parameters)
    data_array = dataframe.values.T

    percentile = parameters['percentile']
    add_per = (100-percentile)/2
    for c in columns:
        # confirmations and exceptions of the pattern, a list of booleans
        upper = round(np.percentile(data_array[c, :], percentile + add_per),2)
        lower = round(np.percentile(data_array[c, :], add_per),2)

        co_up = reduce(operators['<='], [data_array[c, :], upper])
        co_down = reduce(operators['>='], [data_array[c, :], lower])
        co = np.logical_and.reduce([co_up, co_down])

        co_sum, ex_sum, conf = derive_pattern_statistics(co)

        if (conf >= confidence) and (co_sum >= support):
        # confirmations and exceptions of the pattern, a list of booleans
            possible_expression = generate_single_expression([dataframe.columns[c]], [lower, upper], pattern)
            pandas_expressions = to_pandas_expressions(possible_expression, {}, parameters, dataframe)
            xbrl_expressions = to_xbrl_expressions(possible_expression, {}, parameters)
            pattern_data = [[[pattern_name, 0], possible_expression, [co_sum, ex_sum, conf]] + pandas_expressions + xbrl_expressions + ['']]
            yield pattern_data


def patterns_column_column(dataframe  = None,
                           pattern    = None,
                           pattern_name = "column",
                           P_columns  = None,
                           Q_columns  = None,
                           parameters = {}):
    '''Generate patterns of the form [c1] operator [c2] where c1 and c2 are in columns
    '''
    logger = logging.getLogger(__name__)

    confidence, support = get_parameters(parameters)
    decimal = parameters.get("decimal", 0)
    parameters['nonzero'] = True
    initial_data_array = dataframe.values.T
    # set up boolean masks for nonzero items per column
    nonzero = initial_data_array != 0
    count = 0

    preprocess_operator = preprocess[pattern]
    if pattern == "=":
        duplicates = {}
    for c0 in P_columns:
        for c1 in Q_columns:
            count += 1
            if count == 40:
                logger.warning(' More than 40 possibilities!')
            if c0 != c1:
                # applying the filter
                data_filter = reduce(preprocess_operator, [nonzero[c] for c in [c0, c1]])
                if data_filter.any():
                    data_array = initial_data_array[:, data_filter]
                    if data_array.any():
                        if pattern == "=":
                            co = np.abs(data_array[c0, :] - data_array[c1, :]) < 1.5 * 10**(-decimal)
                            if c0 in duplicates:
                                if c1 in duplicates[c0]:
                                    continue
                                else:
                                    if c1 in duplicates:
                                        duplicates[c1].append(c0)
                                    else:
                                        duplicates[c1] = [c0]
                            else:
                                duplicates[c1] = [c0]
                        else:
                            co = reduce(operators[pattern], data_array[[c0, c1], :])
                        co_sum, ex_sum, conf = derive_pattern_statistics(co)
                        if (conf >= confidence) and (co_sum >= support):
                            possible_expression = generate_single_expression([dataframe.columns[c0]], [dataframe.columns[c1]], pattern)
                            pandas_expressions = to_pandas_expressions(possible_expression, {}, parameters, dataframe)
                            xbrl_expressions = to_xbrl_expressions(possible_expression, {}, parameters)
                            pattern_data = [[[pattern_name, 0], possible_expression, [co_sum, ex_sum, conf]] + pandas_expressions + xbrl_expressions + ['']]
                            yield pattern_data

def patterns_sums_column( dataframe  = None,
                         pattern_name = None,
                         P_columns  = None,
                         Q_columns  = None,
                         parameters = {}):
    '''Generate patterns of the form sum [c1-list] = [c2] where c1-list is column list and c2 is column
    '''
    logger = logging.getLogger(__name__)

    confidence, support = get_parameters(parameters)
    sum_elements = parameters.get("sum_elements", 2)
    decimal = parameters.get("decimal", 0)
    initial_data_array = dataframe.values.T
    parameters['nonzero'] = True
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

    count = 0
    neg_col = [1]*len(Q_columns)
    for lhs_elements in range(2, sum_elements + 1):
        for rhs_column in Q_columns:
            start_array = initial_data_array
            # minus righthandside is taken so we can use sum function for all columns
            try:
                start_array[rhs_column, :] = -start_array[rhs_column, :]
                neg_col[rhs_column] = -neg_col[rhs_column]
            except:
                continue
            lhs_column_list = [col for col in P_columns if (col != rhs_column)]
            for lhs_columns in itertools.combinations(lhs_column_list, lhs_elements):
                count += 1
                if count == 50:
                    logger.warning(' More than 50 possibilities!')
                all_columns = lhs_columns + (rhs_column,)
                data_filter = np.logical_and.reduce(nonzero[all_columns, :])
                if data_filter.any():
                    data_array = start_array[:, data_filter]
                    co = (abs(np.sum(data_array[all_columns, :], axis = 0)) < 1.5 * 10**(-decimal))
                    co_sum, ex_sum, conf = derive_pattern_statistics(co)
                    # we only store the patterns that satisfy criteria
                    if (conf >= confidence) and (co_sum >= support):
                        possible_expression = generate_single_expression([dataframe.columns[c] for c in lhs_columns], [dataframe.columns[rhs_column]], 'sum', [neg_col[c] for c in all_columns])
                        pandas_expressions = to_pandas_expressions(possible_expression, {}, parameters, dataframe)
                        xbrl_expressions = to_xbrl_expressions(possible_expression, {}, parameters)
                        pattern_data = [[[pattern_name, 0], possible_expression, [co_sum, ex_sum, conf]] + pandas_expressions + xbrl_expressions + ['']]
                        yield pattern_data

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
    logger = logging.getLogger(__name__)

    logger.debug(' Pattern in: ' + pattern)
    # preprocessing step
    res = preprocess_pattern(pattern, parameters)
    # datapoints to pandas, i.e. {column} -> df[column]
    res, nonzero_col = datapoints2pandas(res, encode)
    # expression to pandas, i.e. IF X=x THEN Y=y -> df[df[X]=x & df[Y]=y] for confirmations
    co_str, ex_str = expression2pandas(res, nonzero_col, parameters)
    logger.debug(' Pandas out: ' + co_str)

    return [co_str, ex_str]

def to_dataframe(patterns = None, parameters = {}, encodings={}):
    '''Convert list of patterns to dataframe with patterns
    '''
    # unpack pattern_id and pattern and patterns_stats and exclude co and ex and set pattern status to unknown
    patterns = list(patterns)
    if len(patterns) > 0:
        data = [pattern_id + [pattern] + pattern_stats + [INITIAL_PATTERN_STATUS] + [encodings] +
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
            try:
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
            except TypeError as e:
                df_patterns.loc[idx, SUPPORT] = None
                df_patterns.loc[idx, EXCEPTIONS] = None
                df_patterns.loc[idx, CONFIDENCE] = None
                df_patterns.loc[idx, ERROR] = e
            except:
                df_patterns.loc[idx, SUPPORT] = None
                df_patterns.loc[idx, EXCEPTIONS] = None
                df_patterns.loc[idx, CONFIDENCE] = None
                df_patterns.loc[idx, ERROR] = 'ERROR unknown'

            df_new_patterns = df_patterns

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
    logger = logging.getLogger(__name__)

    logger.info('Analyze started ...')
    logger.info('Shape of df_patterns: ' + str(df_patterns.shape))

    if (P_dataframe is not None) and (Q_dataframe is not None):
        try:
            dataframe = P_dataframe.join(Q_dataframe)
        except:
            logger.error("Join of P_dataframe and Q_dataframe failed, overlapping columns?")
            return []
    encodings = get_encodings()

    if (dataframe is not None) and (df_patterns is not None):

        df = dataframe.copy()
        results = list()
        for idx in df_patterns.index:
            pandas_ex = df_patterns.loc[idx, PANDAS_EX]
            pandas_co = df_patterns.loc[idx, PANDAS_CO]
            # print(idx)
            encode = df_patterns.loc[idx, ENCODINGS]

            try:
                results_ex = eval(pandas_ex, encodings, {'df': df, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index.values.tolist()
                results_co = eval(pandas_co, encodings, {'df': df, 'MAX': np.maximum, 'MIN': np.minimum, 'SUM': np.sum}).index.values.tolist()
                colq = get_value(df_patterns.loc[idx, "pattern_def"], 2, 1)
                colp = get_value(df_patterns.loc[idx, "pattern_def"], 1, 1)
                if isinstance(colq, list):
                    dataframe['combined_q']= dataframe[colq].values.tolist()
                    colq_old = colq
                    colq = 'combined_q'

                if isinstance(colp, list):
                    dataframe['combined_p']= dataframe[colp].values.tolist()
                    colp = 'combined_p'
                if colp != None:
                    colp = dataframe.columns.get_loc(colp)
                if colq != None:
                    colq = dataframe.columns.get_loc(colq)
                for i in results_ex:
                    k = dataframe.index.get_loc(i)

                    if colp != None:
                        values_p = dataframe.iloc[k, colp]
                        if isinstance(values_p, pd.Series):
                            if len(values_p) > 1:
                                values_p = 'Duplicate indices!'
                            else:
                                values_p = values_p[0]
                    else:
                        values_p = ""
                    if colq != None:

                        values_q = dataframe.iloc[k, colq]
                        if isinstance(values_q, pd.Series):
                            if len(values_q) > 1:
                                values_q = 'Duplicate indices!'
                            else:
                                values_q = values_q[0]
                    else:
                        values_q = ""

                    # Encode if necessary
                    # if encode != {}:
                    #     if not isinstance(values_p, list):
                    #         values_p = [values_p]
                    #     if not isinstance(values_q, list):
                    #         values_q = [values_q]
                    #     for c in range(len(colq_old)):
                    #         if colq_old[c] in encode.keys():
                    #             values_q[c] = eval(str(encode[colq_old[c]])+ "(s)", encodings, {'s': values_q[c]})
                    #     for c in range(len(colq_old)):
                    #         if colq_old[c] in encode.keys():
                    #             values_q[c] = eval(str(encode[colq_old[c]])+ "(s)", encodings, {'s': values_q[c]})

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
                    k = dataframe.index.get_loc(i)
                    if colp != None:
                        values_p = dataframe.iloc[k, colp]
                        if isinstance(values_p, pd.Series):
                            if len(values_p) > 1:
                                values_p = 'Duplicate indices!'
                            else:
                                values_p = values_p[0]
                    else:
                        values_p = ""
                    if colq != None:
                        values_q = dataframe.iloc[k, colq]
                        if isinstance(values_q, pd.Series):
                            if len(values_q) > 1:
                                values_q = 'Duplicate indices!'
                            else:
                                values_q = values_q[0]
                    else:
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
                logger.info("Pattern " + df_patterns.loc[idx, "pattern_id"] + " correctly parsed (#co=" +str(len(results_co)) + ", #ex=" + str(len(results_ex))+")")
            except TypeError as e:
                results.append([True,
                                df_patterns.loc[idx, "pattern_id"],
                                df_patterns.loc[idx, "cluster"],
                                0,
                                df_patterns.loc[idx, "support"],
                                df_patterns.loc[idx, "exceptions"],
                                df_patterns.loc[idx, "confidence"],
                                df_patterns.loc[idx, "pattern_def"],
                                df_patterns.loc[idx, ERROR],
                                'BUG'])
                logger.error('Error in analyze: ' + str(e))
            except:
                results.append([True,
                                df_patterns.loc[idx, "pattern_id"],
                                df_patterns.loc[idx, "cluster"],
                                0,
                                df_patterns.loc[idx, "support"],
                                df_patterns.loc[idx, "exceptions"],
                                df_patterns.loc[idx, "confidence"],
                                df_patterns.loc[idx, "pattern_def"],
                                df_patterns.loc[idx, ERROR],
                                'BUG'])
                logger.error('Error in analyze: UNKOWN')

        df_results = pd.DataFrame(data = results, columns = RESULTS_COLUMNS)
        df_results.sort_values(by = ["index", "confidence", "support"], ascending = [True, False, False], inplace = True)
        df_results.set_index(["index"], inplace = True)
        try:
            df_results.index = pd.MultiIndex.from_tuples(df_results.index)
        except:
            df_results.index = df_results.index

    for level in range(len(dataframe.index.names)):
        del dataframe[dataframe.index.names[level]]
    df_results = ResultDataFrame(df_results)
    logger.info('Shape of df_results: ' + str(df_results.shape))
    logger.info('Analyze ended ...')
    return df_results


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
