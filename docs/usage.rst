=====
Usage
=====

To use data-patterns in a project::

    import data_patterns

The data-patterns package is able to generate and evaluate data-patterns in the content of Pandas DataFrames.

Finding simple patterns
-----------------------

To introduce the features of the this package define the following Pandas DataFrame::

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

Start by defining a PatternMiner::

    miner = data_patterns.PatternMiner(df)

To generate patterns use the find-function of this object::

    df_patterns = miner.find({'name'      : 'equal values',
                              'pattern'   : '=',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 2,
                                             "decimal" : 8}})

The name of the pattern is shown in the output. It is not necessary to include a name.

The result is a DataFrame with the patterns that were found. The first part of the DataFrame now contains

+----+--------------+---------------------------+----------+-----------+----------+
| id |pattern_id    |pattern_def                |support   |exceptions |confidence|
+====+==============+===========================+==========+===========+==========+
|  0 |equal values  | {Own funds} = {Excess}    |9         |1          |0.9       |
+----+--------------+---------------------------+----------+-----------+----------+

The miner finds one patterns; it states that the 'Own funds'-column is identical to the 'Excess'-column in 9 of the 10 cases (with a confidence of 90 %, there is one case where the equal-pattern does not hold).


To analyze data with the generated set of data-patterns use the analyze function::

    df_results = miner.analyze()

The result is a DataFrame with the results. If we select ``result_type = False`` then the first part of the output contains

+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+
|index      |result_type   |pattern_id   |pattern_def                |support   |exceptions |confidence|P values |Q values |
+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+
|Insurer 10 |False         |equal values | {Own funds} = {Excess}    |9         |1          |0.9       |200      |199.99   |
+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+

Note: In case that you have predefined patterns, you need to run it like this

miner = data_patterns.PatternMiner(df_patterns=predefined_patterns)
results = miner.analyze(df)

Other patterns you can use are '>', '<', '<=', '>=', '!=', 'sum' (see below), and '-->' (association, see below).


Using conditional pattern
-------------------------
With the conditional pattern you can find conditional statements between columns, such as IF TV-life = 0 THEN TV-nonlife > 0::

    df_patterns = miner.find({'name'     : 'Pattern 1',
         'pattern'  : '-->',
         'P_columns': ['TV-life'],
         'P_values' : [0],
         'Q_columns': ['TV-nonlife'],
         'Q_values' : [0],
         'parameters' : {"min_confidence" : 0.5, "min_support" : 1,'Q_operators': ['>']}})

results in a DataFrame with

+----+--------------+---------------------------------------------------+----------+-----------+----------+
| id |pattern_id    |pattern_def                                        |support   |exceptions |confidence|
+====+==============+===================================================+==========+===========+==========+
|  0 |equal values  | IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} > 0)    |6         |0          |1.0       |
+----+--------------+---------------------------------------------------+----------+-----------+----------+

The miner finds one condition; apparently the 'TV-life'-column is 6 times 0 and 'TV-nonlife' is then larger than 0.

One can define the values, operators and logics. The values are normally set to none and will then try every possible option for the values. The operators are put in the parameters as shown above and are set to '=' when none are given. Logics are the operators between columns such as '&' and '|' (AND, OR). Logics are also put in the parameters as 'Q_logics' or 'P_logics'. These can only be used when we have more than one column in P or Q. This is set to '&' when none are given. 

An easier approach is to use text for a conditional statement. See the Expression chapter for more information.

One can use the parameter "min_confidence" : "highest" for conditional patterns. In the case that we do not know the values that we are looking for, it will find the most common pattern for each P-value. Let's say we have IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = @), where @ is an unknown value and we get out

+----+--------------+------------------------------------------------------+----------+-----------+----------+
| id |pattern_id    |pattern_def                                           |support   |exceptions |confidence|
+====+==============+======================================================+==========+===========+==========+
|  0 |equal values  | IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = 8800)    |3         |3          |0.5       |
+----+--------------+------------------------------------------------------+----------+-----------+----------+
|  0 |equal values  | IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = 7200)    |1         |5          |0.1667    |
+----+--------------+------------------------------------------------------+----------+-----------+----------+
|  0 |equal values  | IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = 200)     |1         |5          |0.1667    |
+----+--------------+------------------------------------------------------+----------+-----------+----------+

The parameter  "min_confidence" : "highest" would only pick the first pattern and delete the rest.


Using the sum-pattern
---------------------

With the sum-pattern you can find columns whose values are the sum of the values of other columns. For example::

    df_patterns = miner.find({'name'      : 'sum pattern',
                              'pattern'   : 'sum',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 1}})

results in a DataFrame with

+----+--------------+----------------------------------------------------+--------+-----------+----------+
| id |pattern_id    |pattern_def                                         |support |exceptions |confidence|
+====+==============+====================================================+========+===========+==========+
|0   |sum pattern   |({"TV-life"} + {"Own funds"} = {"Assets"})          |4       |0          |1.0       |
+----+--------------+----------------------------------------------------+--------+-----------+----------+
|1   |sum pattern   |({"TV-life"} + {"Excess"} = {"Assets"})             |4       |0          |1.0       |
+----+--------------+----------------------------------------------------+--------+-----------+----------+
|2   |sum pattern   |({"TV-nonlife"} + {"Own funds"} = {"Assets"})       |5       |1          |0.8333    |
+----+--------------+----------------------------------------------------+--------+-----------+----------+
|3   |sum pattern   | ({"TV-nonlife"} + {"Excess"} = {"Assets"})         |5       |1          |0.8333    |
+----+--------------+----------------------------------------------------+--------+-----------+----------+


The miner finds four sums; apparently the 'TV-life'-column plus the 'Own funds'-columns is a sum of the 'Assets'-columns.

With an additional parameter ``sum_elements`` you can specify the highest number of elements in the P_columns. But handle with care because to find a high number of elements can take a lot of time. The default value of ``sum_elements`` is 2.

Using expressions
-----------------

We can also find the same patterns as above using expressions::

    df_patterns = miner.find({'name'      : 'equal values',
                              'expression'   : '{.*}={.*}',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 2}})
                                             
    df_patterns = miner.find({'name'      : 'equal values',
                              'expression'   : 'IF {"TV-life"} = 0 THEN {"TV-nonlife"} > 0',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 2}})
                                             
    df_patterns = miner.find({'name'      : 'sum pattern',
                              'expression'   : '{.*} + {.*} = {.*}', # example {"TV-life"} + {"Own funds"} = {"Assets"}
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 2}})

This will give the same result as the equal, conditional and sum pattern.

Expressions can be written as followed:

1. Put it in a structure like above
2. Columns are given with '{}', example: '{"Assests"} > 0'
3. If you want to find matches with columns you can do '{.*}' (this will match all columns), example: '{.*TV.*} > 0' (will match TV-life and TV-nonlife)
4. Conditional statements go with IF, THEN together with & and | (and/or), example: 'IF ({.*TV-life.*} = 0) THEN ({.*TV-nonlife.*} = 8800) & {.*As.*} > 0)' Note: AND is only used when you want the reverse of this statement, such as 'IF ({.*TV-life.*} = 0) THEN ({.*TV-nonlife.*} = 8800) & {.*As.*} > 0) AND IF ({.*TV-life.*} = 0) THEN ~({.*TV-nonlife.*} = 8800) & {.*As.*} > 0)'
5. Use [@] if you do not have a specific value, example: 'IF ({.*Ty.*} = [@]) THEN ({.*As.*} = [@])'. This will match all the possible values of 'Type' and 'Assets' in the pattern, such as 'IF ({'Type'} = 'life-insurer') THEN ({'Assets'} = 1000)', etc
6. You can also use regex in front of @ such as ({.*Ty.*} = ["non-life"@]), which would only match 'non-life insurer'
7. you can also use multiple columns: 'IF ({.*TV-life.*,TV-nonlife.*} = 0)'. This would match both columns.

Summary of patterns
--------------------

There are two categories:

1. conditional: IF THEN

2. Quantitative: Sum, col_a = col_b, col_a > value

One can also do more complex quantitative patterns such as: '{"Col_A"}*{"Col_B"}/{"Col_C"}={"Col_D"}. However on must use the parameter 'expres':True for this to work. It can only be used when it is directly used as a pandas expression to find patterns and that is why you need the parameter.

You can also use the following in expressions: MAX, MIN, SUM, ABS. Such as ABS({"Col_A"}-{"Col_B"}) = {"Col_C"}.

Setting the parameters dict
---------------------------

Specific parameters of a pattern can be set with a parameters dict. ``min_confidence`` defines the minimum confidence of the patterns to be included in the output and ``min_support`` defines the minimum support of the patterns. 

For the =-patterns, you can set the number of decimals for the equality between the values with ``decimal``. So::

    df_patterns = miner.find({'name'      : 'equal values',
                              'pattern'   : '=',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 2,
                                             "decimal"       : 0}})

would output

+----+--------------+---------------------------+----------+-----------+----------+
| id |pattern_id    |pattern_def                |support   |exceptions |confidence|
+====+==============+===========================+==========+===========+==========+
|  0 |equal values  | {Own funds} = {Excess}    |10        |0          |1.0       |
+----+--------------+---------------------------+----------+-----------+----------+

because 199.99 is equal to 200 with 0 decimals.

The default value in the =-pattern is 0 decimals.

You do not have to include a paramaters dict. The parameters have default setting with ``min_confidence = 0.75`` and ``min_support = 2``.

A list of parameters that you can use:

    - min_confidence (0.75): float [0,1] or 'highest'
    - min_support(2): int > 0
    - decimal (0): int. Used for rounding and comparing values to that decimal. Can be negative and then values will be rounded to the nearest 10th (for -1), 100th for (-2), etc 
    - window (None): int. Only compare columns in that window
    - disable (False): boolean. Disables tqdm bars
    - expres (Flase): boolean. Uses pandas expression to find patterns and does not use the numpy dissection (needed for quant patterns not following the standerd format)
    - nonNaN (False): boolean. Ignores NaN values, otherwise one can have patterns with NaN values taken into account.
    - nonzero(Flase): boolean. Ignores 0 values
    

Convert columns to time
-----------------------

We also wanted to look at time patterns. Therefore we had to transform tables into another format. We have two possible formats:

1. convert_to_time():

This column matches names and consecutive periodes into one row with a prefix

+------+-----------+--------------+
| Name | periode   |Assets        | 
+======+===========+==============+
|Ins1  |2018       |1000          |
+------+-----------+--------------+
|Ins2  |2018       |2000          |
+------+-----------+--------------+
|Ins1  |2019       |1500          |
+------+-----------+--------------+
|Ins2  |2019       | 2015         |
+------+-----------+--------------+

will transform into

+------+-----------+--------------+-------------+
| Name | periode   |Assets (t-1)  | Assets (t)  | 
+======+===========+==============+=============+
|Ins1  |2018-2019  |1000          | 1500        |
+------+-----------+--------------+-------------+
|Ins2  |2018-2019  |2000          | 2015        |
+------+-----------+--------------+-------------+

You can set set_year=False if you are note dealing with whole years.

2. convert_columns_to_time():

+------+-----------+--------------+
| Name | periode   |Assets        | 
+======+===========+==============+
|Ins1  |2018       |1000          |
+------+-----------+--------------+
|Ins2  |2018       |2000          |
+------+-----------+--------------+
|Ins1  |2019       |1500          |
+------+-----------+--------------+
|Ins2  |2019       | 2015         |
+------+-----------+--------------+

will transform into

+------+-----------+--------------+-------------+
| Name | Datapoint |2018          | 2019        | 
+======+===========+==============+=============+
|Ins1  |Assets     |1000          | 1500        |
+------+-----------+--------------+-------------+
|Ins2  |Assets     |2000          | 2015        |
+------+-----------+--------------+-------------+


Correct data
------------
Sometimes you want to correct data, such as

+------+-----------+
| LEI  | Name      |
+======+===========+
|1005  |BANK       |
+------+-----------+
|1005  |BANK       |
+------+-----------+
|1005  |Bank       |
+------+-----------+
 where you want all the names to be the same for the same code. If you perform a simple conditional pattern with conf 'highest', you can use the function correct_data() to change the names.


Finding a list of patterns
--------------------------

You can start the find-function with a dictionary (with one pattern definition) or a list of dictionaries (with a list of pattern definitions).


Applying encodings
------------------

You might wish to apply to encode one or more columns before generating data-patterns. You can specify a ``encode`` in the definition dict of the pattern::

    p = {'name'     : 'Pattern 1',
         'pattern'  : '-->',
         'P_columns': ['Type'],
         'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
         'encode'   : {'Assets'   : 'reported',
                      'TV-life'   : 'reported',
                      'TV-nonlife': 'reported',
                      'Own funds' : 'reported'}}
    miner = data_patterns.PatternMiner(p)

The function ``reported`` is a simple function that returns "not reported" if the value is nan or zero and "reported" otherwise. 

This pattern-definition finds conditional patterns ('-->') between 'Type' and whether the columns 'Assets', 'TV-life', 'TV-nonlife', 'Own funds' are reported or not.

+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------+----------+
| id |pattern_id    |pattern_def                                                                                                                                                            |support |exceptions |confidence|
+====+==============+=======================================================================================================================================================================+========+===========+==========+
|0   |Pattern 1     |IF ({"Type"} = "life insurer") THEN ({"Assets"} = "reported") & ({"TV-life"} = "reported") & ({"TV-nonlife"} = "not reported") & ({"Own funds"} = "reported")          |4       |1          |0.8       |
+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------+----------+
|1   |Pattern 1     |IF ({"Type"} = "non-life insurer") THEN ({"Assets"} = "reported") & ({"TV-life"} = "not reported") & ({"TV-nonlife"} = "reported") & ({"Own funds"} = "reported")      |5       |0          |1.0       |
+----+--------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+-----------+----------+

So the pattern is that life insurers report Assets, TV-life, and Own funds and nonlife insurers report Assets, TV-nonlife and Own funds. There is one life insurer that does not report according to these patterns.


Retrieving the pattern in Pandas
--------------------------------

The df_patterns-dataframe contains the code of the pattern in Pandas::

    df_patterns.loc[0, 'pandas co']

results in the following string::

    df[(df["Type"]=="life insurer") & ((reported(df["Assets"])=="reported") &
    (reported(df["Own funds"])=="reported") &
    (reported(df["TV-life"])=="reported") &
    (reported(df["TV-nonlife"])=="not reported"))]

The code creates a boolean mask based on the pattern and returns the dataframe with data for which the pattern holds.

Similarly, you can find the exceptions of a pattern with::

    df_patterns.loc[0, 'pandas ex']



We plan to provide codings of the pattern based on other relevant packages.
