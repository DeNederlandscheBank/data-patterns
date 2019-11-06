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
                                             "min_support"   : 2}})

The name of the pattern is shown in the output. It is not necessary to include a name.

The result is a DataFrame with the patterns that were found. The first part of the DataFrame now contains

+----+--------------+------------+--------------+------------+--------+-----------+----------+
| id |pattern_id    |P columns   |relation type |Q columns   |support |exceptions |confidence|
+====+==============+============+==============+============+========+===========+==========+
|  0 |equal values  |[Own funds] |=             |[Excess]    |9       |1          |0.9       |
+----+--------------+------------+--------------+------------+--------+-----------+----------+
|  1 |equal values  |[Excess]    |=             |[Own funds] |9       |1          |0.9       | 
+----+--------------+------------+--------------+------------+--------+-----------+----------+

The miner finds two patterns; the first states that the 'Own funds'-column is identical to the 'Excess'-column in 9 of the 10 cases (with a confidence of 90 %, there is one case where the equal-pattern does not hold), and the second pattern is identical to the first but with the columns reversed.

To analyze data with the generated set of data-patterns use the analyze function with the dataframe with the data as input::

    df_results = miner.analyze(df)

The result is a DataFrame with the results. If we select ``result_type = False`` then the first part of the output contains

+-----------+--------------+-------------+------------+-------------+------------+---------+---------+
|index      |result_type   |pattern_id   |P columns   |relation type|Q columns   |P values |Q values |
+-----------+--------------+-------------+------------+-------------+------------+---------+---------+
|Insurer 10 |False         |equal values |[Own funds] |=            |[Excess]    |[200]    |[199.99] |
+-----------+--------------+-------------+------------+-------------+------------+---------+---------+
|Insurer 10 |False         |equal values |[Excess]    |=            |[Own funds] |[199.99] |[200]    |
+-----------+--------------+-------------+------------+-------------+------------+---------+---------+

Other patterns you can use are '>', '<', '<=', '>=', '!=', 'sum' (see below), and '-->' (association, see below).

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

+----+--------------+------------+--------------+------------+--------+-----------+----------+
| id |pattern_id    |P columns   |relation type |Q columns   |support |exceptions |confidence|
+====+==============+============+==============+============+========+===========+==========+
|  0 |equal values  |[Own funds] |=             |[Excess]    |10      |0          |1.0       |
+----+--------------+------------+--------------+------------+--------+-----------+----------+
|  1 |equal values  |[Excess]    |=             |[Own funds] |10      |0          |1.0       | 
+----+--------------+------------+--------------+------------+--------+-----------+----------+

because 199.99 is equal to 200 with 0 decimals.

The default value in the =-pattern is 8 decimals.

You do not have to include a paramaters dict. The parameters have default setting with ``min_confidence = 0.75`` and ``min_support = 2``.

Using the sum-pattern
---------------------

With the sum-pattern you can find columns whose values are the sum of the values of other columns. For example::

    df_patterns = miner.find({'name'      : 'sum pattern',
                              'pattern'   : 'sum',
                              'parameters': {"min_confidence": 0.5,
                                             "min_support"   : 1}})

results in a DataFrame with

+----+--------------+------------------------+--------------+------------+--------+-----------+----------+
| id |pattern_id    |P columns               |relation type |Q columns   |support |exceptions |confidence|
+====+==============+========================+==============+============+========+===========+==========+
|0   |sum pattern   |[TV-life, Own funds]    |sum           |[Assets]    |5       |0          |1.0       |
+----+--------------+------------------------+--------------+------------+--------+-----------+----------+
|1   |sum pattern   |[TV-life, Excess]       |sum           |[Assets]    |5       |0          |1.0       |
+----+--------------+------------------------+--------------+------------+--------+-----------+----------+
|2   |sum pattern   |[TV-nonlife, Own funds] |sum           |[Assets]    |4       |1          |0.8       |
+----+--------------+------------------------+--------------+------------+--------+-----------+----------+
|3   |sum pattern   |[TV-nonlife, Excess]    |sum           |[Assets]    |4       |1          |0.8       |
+----+--------------+------------------------+--------------+------------+--------+-----------+----------+

The miner finds four sums; apparently the 'TV-life'-column plus the 'Own funds'-columns is a sum of the 'Assets'-columns.

With an additional parameter ``sum_elements`` you can specify the highest number of elements in the P_columns. But handle with care because to find a high number of elements can take a lot of time. The default value of ``sum_elements`` is 2.

Finding a list of patterns
--------------------------

You can start the find-function with a dictionary (with one pattern definition) or a list of dictionaries (with a list of pattern definitions).

Exporting to and importing from Excel
-------------------------------------

You can export the df_patterns to Excel with::

    df_patterns.to_excel(filename = "export.xlsx")

The DataFrame contains a specialized function to generate a humanly readable format of the patterns.

You can import a df_pattern from Excel with::

    miner = data_patterns.PatternMiner(df_patterns = data_patterns.read_excel(filename = "export.xlsx"))

Applying encodings
------------------

You might wish to apply to encode one or more columns before generating data-patterns. You can specify a ``encode`` in the definition dict of the pattern::

    p = {'name'     : 'Pattern 1',
         'pattern'  : '-->',
         'P_columns': ['Type'],
         'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
         'encode'   : {'Assets'   : data_patterns.reported,
                      'TV-life'   : data_patterns.reported,
                      'TV-nonlife': data_patterns.reported,
                      'Own funds' : data_patterns.reported}}
    miner = data_patterns.PatternMiner(p)

The function ``data_patterns.reported`` is a simply function that returns "not reported" if the value is nan or zero and "reported" otherwise. (TO DO: using user defined encode-functions)

This pattern-definition finds association patterns ('-->') between 'Type' and whether the columns 'Assets', 'TV-life', 'TV-nonlife', 'Own funds' are reported or not.

+----+-----------+-------------------+---------+---------------------------------------------+--------+-----------+----------+
| id |pattern_id |P                  |relation |Q                                            |support |exceptions |confidence|
+====+===========+===================+=========+=============================================+========+===========+==========+
|  0 |Pattern 1  |[life insurer]     |-->      |[reported, reported, reported, not reported] |4       |1          |0.8       |
+----+-----------+-------------------+---------+---------------------------------------------+--------+-----------+----------+
|  1 |Pattern 1  |[non-life insurer] |-->      |[reported, reported, not reported, reported] |5       |0          |1.0       |
+----+-----------+-------------------+---------+---------------------------------------------+--------+-----------+----------+

So the pattern is that life insurers report Assets, TV-life, and Own funds and nonlife insurers report Assets, TV-nonlife and Own funds. There is one life insurer that does not report according to these patterns.

Retrieving the pattern in Pandas-code
-------------------------------------

The df_patterns-dataframe contains the code of the pattern in Pandas::

    df_patterns.loc[0, 'pandas co']

results in::

    'df[(df["Type"]=="life insurer") & ((data_patterns.reported(df["Assets"])=="reported") & (data_patterns.reported(df["Own funds"])=="reported") & (data_patterns.reported(df["TV-life"])=="reported") & (data_patterns.reported(df["TV-nonlife"])=="not reported"))]' 

The code creates a boolean mask based on the pattern and returns the dataframe with data for which the pattern holds.

Similarly, you can find the exceptions of a pattern with::

    df_patterns.loc[0, 'pandas ex']

We plan to provide codings of the pattern based on other relevant packages.

