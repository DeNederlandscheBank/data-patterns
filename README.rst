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
.. image:: https://img.shields.io/badge/License-MIT/X-blue.svg
        :target: https://github.com/DeNederlandscheBank/data-patterns/blob/master/LICENSE
        :alt: License

Package for generating and evaluating data-patterns in quantitative reports

* Free software: MIT/X license
* Documentation: https://data-patterns.readthedocs.io.


Features
--------

Here is what the package does:

- Generating and evaluating patterns in structured datasets and exporting to Excel and JSON
- Transforming generated patterns into Pandas code

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
                                             "min_support"   : 2,
                                             "decimal" : 8}})

The result is a DataFrame with the patterns that were found. The first part of the DataFrame now contains

+----+--------------+---------------------------+----------+-----------+----------+
| id |pattern_id    |pattern_def                |support   |exceptions |confidence|
+====+==============+===========================+==========+===========+==========+
|  0 |equal values  | {Own funds} = {Excess}    |9         |1          |0.9       |
+----+--------------+---------------------------+----------+-----------+----------+


The miner finds one patterns; it states that the 'Own funds'-column is identical to the 'Excess'-column in 9 of the 10 cases (with a confidence of 90 %, there is one case where the equal-pattern does not hold).


To analyze data with the generated set of data-patterns use the analyze function with the dataframe with the data as input::

    df_results = miner.analyze(df)

The result is a DataFrame with the results. If we select ``result_type = False`` then the first part of the output contains

+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+
|index      |result_type   |pattern_id   |pattern_def                |support   |exceptions |confidence|P values |Q values |
+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+
|Insurer 10 |False         |equal values | {Own funds} = {Excess}    |9         |1          |0.9       |200      |199.99   |
+-----------+--------------+-------------+---------------------------+----------+-----------+----------+---------+---------+

Other patterns you can use are '>', '<', '<=', '>=', '!=', 'sum', and '-->'.

Read the documentation for more features.

Upload to Pypi (for developers)
-------------------------------

1. Change the version in setup.py
2. Go to github.com and navigate to the repository. Next, click on the tab “releases” and then on “Create a new release”. Now, define a Tag verion (it is best to use the same number as you used in your setup.py version-field: v0.1.15 for example). Then click on “publish release”.
3. Make a Pypi account here: https://pypi.org/manage/projects/
4. Download twine by typing in your command prompt::

    pip install twine

5. Get admin rights of the owner of the data_patterns package.
6. Open your command prompt and go to the folder of data_patterns. Then type

    python setup.py sdist

    twine upload dist/*

A good reference is here: https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
