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
                                3, 1, 0.75]])
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
        p2 = {'name'      : 'Pattern 1',
            'expression' : 'IF ({.*Ty.*} = [@]) THEN ({.*.*} = [@])'}
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
        self.assertEqual(type(actual), type(expected), "Pattern test 7: types do not match")
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
        self.assertEqual(type(actual), type(expected), "Pattern test 8: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern9(self):
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
        p2 = {'name'     : 'Pattern 1', 'cluster':'Type',
                 'pattern'  : '='}
        # Expected output
        expected = pd.DataFrame(columns = ['index','pattern_id', 'cluster', 'pattern_def', 'support', 'exceptions',
                                    'confidence'],
                                data = [[0,'Pattern 1', 'life insurer', '({"Own funds"} = {"Excess"})',
                                5,0,1.0],
                                [1,'Pattern 1', 'non-life insurer', '({"Own funds"} = {"Excess"})',
                                5,0,1.0]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])
        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 9: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern10(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurerx', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        df.set_index('Name', inplace = True)
        p2 = {'name'     : 'Pattern 1', 'expression':'IF {.*TV-l.*} =[@] THEN {.*Typ.*}= [@]'}

        # Expected output
        expected = pd.DataFrame(columns = ['Name',       'Type',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
                  data   = [['Insurer  1', 'life insurer',     1000,     800,       0,             200,         200],
                            ['Insurer  2', 'non-life insurer', 4000,     0,         3200,          800,         800],
                            ['Insurer  3', 'non-life insurer', 800,      0,         700,           100,         100],
                            ['Insurer  4', 'life insurer',     2500,     1800,      0,             700,         700],
                            ['Insurer  5', 'non-life insurer', 2100,     0,         2200,          200,         200],
                            ['Insurer  6', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  7', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  8', 'life insurer',     9000,     8800,      0,             200,         200],
                            ['Insurer  9', 'life insurer', 9000,     8800,      0,             200,         200],
                            ['Insurer 10', 'non-life insurer', 9000,     0,         8800,          200,         199.99]])
        expected.set_index('Name', inplace = True)
        # expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        df_ana = p.analyze()
        actual = p.correct_data()
        # Assert
        self.assertEqual(type(actual[0]), type(expected), "Pattern test 10: types do not match")

        pd.testing.assert_frame_equal(actual[0], expected)

    def test_pattern11(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'periode',             'Assets' ],
                  data   = [['Insurer  1', 2018,     1000    ],
                            ['Insurer  2', 2018, 4000     ],
                            ['Insurer  1', 2019, 800    ],
                            ['Insurer  2', 2019,     2500]])
        miner = data_patterns.PatternMiner(df)
        df_patterns = miner.convert_columns_to_time('Name','periode')
        actual = df_patterns.reset_index()
        # Expected output
        expected = pd.DataFrame(columns = ['Name', 'Datapoint', 2018, 2019],
                                data = [['Insurer  1', 'Assets' ,1000 ,800],
                                ['Insurer  2', 'Assets', 4000 ,2500]])


        # Assert

        self.assertEqual(type(actual), type(expected), "Pattern test 11: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern12(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'periode',             'Assets', 'TV-life', 'TV-nonlife' , 'Own funds', 'Excess'],
              data   = [['Insurer  1', 2018,     1000,     800,       0,             200,         200],
                        ['Insurer  2', 2018, 4000,     0,         3200,          800,         800],
                        ['Insurer  1', 2019, 800,      0,         700,           100,         100],
                        ['Insurer  2', 2019,     2500,     1800,      0,             700,         700]])
        df['periode'] = pd.to_datetime(df['periode'],format='%Y')
        miner = data_patterns.PatternMiner(df)
        df_patterns = miner.convert_to_time(['Name'],'periode')
        actual = df_patterns.reset_index()
        # Expected output
        expected = pd.DataFrame(columns = ['periode', 'Name','Assets (t-1)', 'TV-life (t-1)', 'TV-nonlife (t-1)', 'Own funds (t-1)',
       'Excess (t-1)', 'Assets (t)', 'TV-life (t)', 'TV-nonlife (t)',
       'Own funds (t)', 'Excess (t)'],
                                data = [['2018 - 2019', 'Insurer  1', 1000 ,800 ,0 ,200 ,200, 800, 0 ,700, 100 ,100],
                                ['2018 - 2019' ,'Insurer  2' ,4000 ,0 ,3200 ,800 ,800 ,2500, 1800, 0 ,700 ,700]])


        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 12: types do not match")
        pd.testing.assert_frame_equal(actual, expected)

    def test_pattern13(self):
        """Test of read input date function"""
        # Input
        df = pd.DataFrame(columns = ['Name',       'periode',             'Assets' ],
          data   = [['Insurer  1', 2018,     0    ],
                    ['Insurer  2', 2018, 10     ],
                    ['Insurer  1', 2019, 0    ],
                    ['Insurer  2', 2019,     10]])
        p2 = {'name'     : 'Pattern 1', 'expression':'IF {.*Name.*} =[@] THEN {.*As.*}= [@]'}
        # Expected output
        expected = pd.DataFrame(columns = ['index','result_type', 'pattern_id', 'cluster', 'support', 'exceptions',
       'confidence', 'pattern_def', 'P values', 'Q values'],
                                data = [[0,True, 'Pattern 1', 0 ,2 ,0, 1.0,
                                'IF {"Name"} ="Insurer  1" THEN {"Assets"}= 0' ,'Insurer  1', 0],
                                [1,True ,'Pattern 1', 0, 2 ,0 ,1.0,
                                'IF {"Name"} ="Insurer  2" THEN {"Assets"}= 10' ,'Insurer  2', 10],
                                [2,True ,'Pattern 1', 0 ,2 ,0, 1.0,
                                'IF {"Name"} ="Insurer  1" THEN {"Assets"}= 0' ,'Insurer  1', 0],
                                [3,True, 'Pattern 1' ,0 ,2, 0, 1.0,
                                'IF {"Name"} ="Insurer  2" THEN {"Assets"}= 10', 'Insurer  2', 10]])
        expected.set_index('index', inplace = True)

        expected = data_patterns.ResultDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(df)
        actual = p.find(p2)
        actual = p.analyze()

        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 9: types do not match")
        pd.testing.assert_frame_equal(actual, expected)
