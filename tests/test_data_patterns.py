#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `data_patterns` package."""


import unittest
import os
from data_patterns import data_patterns
import pandas as pd


class TestData_patterns(unittest.TestCase):
    """Tests for `data_patterns` package."""

    def test_pattern1(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        pattern = {'name'     : 'Pattern 1',
                    'pattern' : '-->',
                   'P_columns': ['Type'],
                   'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
                   'encode'   : {'Assets':      'reported',
                                 'TV-life':     'reported',
                                 'TV-nonlife':  'reported',
                                 'Own funds':   'reported'}}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
        'confidence'],
                                data = [[0,'Pattern 1', 0, 'IF ({"Type"} = "life insurer") THEN ({"Assets"} = "reported") & ({"TV-life"} = "reported") & ({"TV-nonlife"} = "not reported") & ({"Own funds"} = "reported")',
                                5, 0, 1],
                                        [1,'Pattern 1', 0, 'IF ({"Type"} = "non-life insurer") THEN ({"Assets"} = "reported") & ({"TV-life"} = "not reported") & ({"TV-nonlife"} = "reported") & ({"Own funds"} = "reported")',
                                        4, 1, 0.8]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(pattern)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])

        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 1: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern2(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        pattern = {'name' : 'Pattern 1',
             'pattern'  : '-->',
             'P_columns': ['TV-life', 'Assets'],
             'P_values' : [100,0],
             'Q_values' : [0,0],
             'Q_columns': ['TV-nonlife', 'Own funds'],
             'parameters' : {"min_confidence" : 0, "min_support" : 1, 'Q_operators': ['>', '>'],
             'P_operators':['<','>'], 'Q_logics':['|'], 'both_ways':False}}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 0, 'IF ({"TV-life"} < 100) & ({"Assets"} > 0) THEN ({"TV-nonlife"} > 0) | ({"Own funds"} > 0)',
                                4, 0, 1.0]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(pattern)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 2: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern3(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        pattern = {'name'      : 'equal values',
                                  'pattern'   : '=',
                                  'value' : 0,
                                  'parameters': {"min_confidence": 0.5,
                                                 "min_support"   : 2}}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'equal values', 0, '({"TV-nonlife"} = 0)',
                                6, 4, .6]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(pattern)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 3: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern4(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        pattern = {'name'     : 'Pattern 1',
             'pattern'  : '-->',
             'P_columns': ['TV-life'],
             'P_values' : [0],
             'Q_columns': ['TV-nonlife'],
             'Q_values' : [8800],
             'parameters' : {"min_confidence" : 0, "min_support" : 1, 'both_ways':True}}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 0, 'IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = 8800) AND IF ~({"TV-life"} = 0) THEN ~({"TV-nonlife"} = 8800)',
                                7, 3, 0.7]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(pattern)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 4: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern5(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',   'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        pattern ={'name'      : 'sum pattern',
                                  'pattern'   : 'sum',
                                  'parameters': {"min_confidence": 0.5,
                                                 "min_support"   : 1,
                                                 "nonzero" : True }}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'sum pattern', 0, '({"TV-life"} + {"Own funds"} = {"Assets"})',
                                6, 0, 1.0],
                                [1,'sum pattern', 0, '({"TV-life"} + {"Excess"} = {"Assets"})',
                                6, 0, 1.0],
                                [2,'sum pattern', 0, '({"TV-nonlife"} + {"Own funds"} = {"Assets"})',
                                3, 1, 0.75],
                                [3,'sum pattern', 0, '({"TV-nonlife"} + {"Excess"} = {"Assets"})',
                                2, 2, 0.50]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(pattern)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 5: types do not match")
        pd.testing.assert_frame_equal(actual, expected)
    def test_pattern6(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        parameters = {'min_confidence': 0.5,'min_support'   : 2}
        p2 = {'name'      : 'Pattern 1',
              'expression' : 'IF ({.*TV-life.*} = 0) THEN ({.*TV-nonlife.*} = 8800) AND IF ~({.*TV-life.*} = 0) THEN ~({.*TV-nonlife.*} = 8800)',
              'parameters' : parameters }
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 0, 'IF ({"TV-life"} = 0) THEN ({"TV-nonlife"} = 8800) AND IF ~({"TV-life"} = 0) THEN ~({"TV-nonlife"} = 8800)',
                                7, 3, 0.7]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 4: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern7(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        parameters = {'min_confidence': 0.5,'min_support'   : 2}
        p2 = {'name'      : 'Pattern 1',
            'expression' : 'IF ({.*Ty.*} = "@") THEN ({.*.*} = "@")'}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 0, 'IF ({"Type"} = "non-life insurer") THEN ({"TV-life"} = 0)',
                                4, 1, 0.8],
                                [1,'Pattern 1', 0, 'IF ({"Type"} = "life insurer") THEN ({"TV-nonlife"} = 0)',
                                5, 0, 1.0],
                                [2,'Pattern 1', 0, 'IF ({"Type"} = "life insurer") THEN ({"Own funds"} = 200)',
                                4, 1, 0.8],
                                [3,'Pattern 1', 0, 'IF ({"Type"} = "life insurer") THEN ({"Excess"} = 200.0)',
                                4, 1, 0.8]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 4: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern8(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'non-life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        parameters = {'min_confidence': 0.3,'min_support'   : 1, 'percentile' : 90}
        p2 = {'name'      : 'Pattern 1',
            'pattern' : 'percentile',
            'columns' : [ 'TV-nonlife', 'Own funds'],
          'parameters':parameters}

        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 0, '({"TV-nonlife"} >= 0.0) & ({"TV-nonlife"} <= 6280.0)',
                                9, 1, 0.9],
                                [1,'Pattern 1', 0, '({"Own funds"} >= 145.0) & ({"Own funds"} <= 755.0)',
                                8, 2, 0.8]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 4: types do not match")
        pd.testing.assert_frame_equal(actual, expected)
