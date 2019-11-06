#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `data_patterns` package."""


import unittest

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
                   'P_columns': ['Type'],
                   'Q_columns': ['Assets', 'TV-life', 'TV-nonlife', 'Own funds'],
                   'encode'   : {'Assets':      data_patterns.reported,
                                 'TV-life':     data_patterns.reported,
                                 'TV-nonlife':  data_patterns.reported,
                                 'Own funds':   data_patterns.reported}}
        # Expected output
        expected = pd.DataFrame(columns = ['index', 'pattern_id', 'cluster', 'P columns', 'relation type', 'Q columns', 'P', 'relation',
                                           'Q', 'support', 'exceptions', 'confidence'],
                                data = [[0, 'Pattern 1', 0, ['Type'], '-->', ['Assets', 'Own funds', 'TV-life', 'TV-nonlife'],
                                        ['life insurer'], '-->', ['reported', 'reported', 'reported', 'not reported'], 5, 0, 1.0],
                                        [1, 'Pattern 1', 0, ['Type'], '-->', ['Assets', 'Own funds', 'TV-life', 'TV-nonlife'],
                                        ['non-life insurer'], '-->', ['reported', 'reported', 'not reported', 'reported'], 4, 1, 0.8]])
        expected.set_index('index', inplace = True)
        expected = data_patterns.PatternDataFrame(expected)

        # Actual output
        p = data_patterns.PatternMiner(pattern)
        actual = p.find(df)
        actual = data_patterns.PatternDataFrame(actual.loc[:, 'pattern_id': 'confidence'])

        # Assert
        self.assertEqual(type(actual), type(expected), "Pattern test 1: types do not match")
        pd.testing.assert_frame_equal(actual, expected)
