=============
data-patterns
=============


.. image:: https://img.shields.io/pypi/v/data_patterns.svg
        :target: https://pypi.python.org/pypi/data_patterns
        :alt: Pypi Version
.. image:: https://img.shields.io/travis/DeNederlandscheBank/data-patterns.svg
        :target: https://travis-ci.org/DeNederlandscheBank/data-patterns
        :alt: Build Status
.. image:: https://readthedocs.org/projects/data-patterns/badge/?version=latest
        :target: https://data-patterns.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Package for generating and evaluating data-patterns in quantitative reports

* Free software: MIT/X license
* Documentation: https://data-patterns.readthedocs.io.


Features
--------

Here is what the package does:

- Generating and evaluating patterns in structured datasets and exporting them to Excel and JSON
- Evaluating reporting data with data quality rules published by De Nederlandsche Bank (to be provided)

Quick overview
--------------

To install the package

::

    pip install data_patterns
    

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

Other patterns you can use are '>', '<', '<=', '>=', '!=', 'sum', and '-->'. 

Read the documentation for more features.

