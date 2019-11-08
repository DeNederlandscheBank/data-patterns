'''Dataframe definitions
'''

#imports
import pandas as pd
import numpy as np
import xlsxwriter
import ast
from .constants import *
from .constants import *

__author__ = """De Nederlandsche Bank"""
__email__ = 'ECDB_berichten@dnb.nl'
__version__ = '0.1.6'

class PatternDataFrame(pd.DataFrame):
    ''' 
    A PatternDataframe is a subclass of a Pandas DataFrame for patterns with 
    a specialized to_excel function to get a readable format

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    See Also
    --------

    Notes
    -----

    '''

    def to_excel(self, filename, *args, **kwargs):
        writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
        sheet_name = kwargs.pop('sheet_name', DEFAULT_SHEET_NAME_PATTERNS)
        font = writer.book.add_format({'font_name': 'Arial', 
                                       'font_size': 10, 
                                       'valign'   : 'top', 
                                       'align'    : 'left', 
                                       'text_wrap': True})
        if len(self.index) > 0:
            df = super(PatternDataFrame, self).copy()
            # make sure that the '='-sign is read properly by Excel
            df[RELATION_TYPE] = "'" + df[RELATION_TYPE]
            df.to_excel(writer, 
                        sheet_name = sheet_name, 
                        merge_cells = False, *args, **kwargs)
        else:
            print("Empty patterns dataframe. No patterns to export.")
        for name in writer.sheets:
            worksheet = writer.sheets[name]
            worksheet.set_column('A:O', None, font)
            worksheet.set_default_row(60)
            worksheet.set_column(1, 1, 20)
            worksheet.set_column(2, 2, 5)
            worksheet.set_column(3, 3, 40)
            worksheet.set_column(4, 4, 5)
            worksheet.set_column(5, 5, 40)
            worksheet.set_column(6, 6, 40)
            worksheet.set_column(7, 7, 5)
            worksheet.set_column(8, 8, 40)
            worksheet.set_column(9, 9, 13)
            worksheet.set_column(10, 10, 13)
            worksheet.set_column(11, 11, 13)
            worksheet.set_column(12, 12, 13)
            worksheet.set_column(13, 13, 25)
            worksheet.set_column(14, 14, 25)
        writer.save()
        writer.close()
        return None

class ResultDataFrame(pd.DataFrame):
    ''' 
    A ResultDataframe is a subclass of a Pandas DataFrame for patterns results with 
    a specialized to_excel function to get a readable format

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    See Also
    --------

    Notes
    -----

    '''

    def to_excel(self, filename, *args, **kwargs):
        writer = pd.ExcelWriter(filename, engine = 'xlsxwriter')
        font = writer.book.add_format({'font_name': 'Arial', 
                                       'font_size': 10, 
                                       'valign'   : 'top', 
                                       'align'    : 'left', 
                                       'text_wrap': True})
        if len(self.index) > 0:
            for pattern_id in self[PATTERN_ID].unique():
                co = self[(self[PATTERN_ID]==pattern_id) & (self[RESULT_TYPE]==TEXT_CONFIRMATION)]
                co = co.drop([PATTERN_ID, RESULT_TYPE], axis = 1)
                ex = self[(self[PATTERN_ID]==pattern_id) & (self[RESULT_TYPE]==TEXT_EXCEPTION)]
                ex = ex.drop([PATTERN_ID, RESULT_TYPE], axis = 1)
                if len(co.index) > 0:
                    co.to_excel(writer, sheet_name = pattern_id + SHEET_NAME_POST_CO, merge_cells = False)
                if len(ex.index) > 0:
                    ex.to_excel(writer, sheet_name = pattern_id + SHEET_NAME_POST_EX, merge_cells = False)
        for name in writer.sheets:
            worksheet = writer.sheets[name]
            worksheet.set_column('A:O', None, font)
            worksheet.set_default_row(60)
            levels = self.index.nlevels
            for n in range(0, levels):
                worksheet.set_column(n, n, 20, font)
            worksheet.set_column(levels-1, levels-1, 5)
            worksheet.set_column(levels, levels, 7)
            worksheet.set_column(levels+1, levels+1, 7)
            worksheet.set_column(levels+2, levels+2, 7)
            worksheet.set_column(levels+3, levels+3, 7)
            worksheet.set_column(levels+4, levels+4, 40)
            worksheet.set_column(levels+5, levels+5, 10)
            worksheet.set_column(levels+6, levels+6, 40)
            worksheet.set_column(levels+7, levels+7, 40)
            worksheet.set_column(levels+8, levels+8, 10)
            worksheet.set_column(levels+9, levels+9, 40)
            worksheet.set_column(levels+10, levels+10, 40)
            worksheet.set_column(levels+11, levels+11, 10)
            worksheet.set_column(levels+12, levels+12, 40)
        writer.save()
        writer.close()
        return None

    # obsolete function
    def to_excel_old(self, path, filename, format = "separate"):
        if format == "separate":
            # level 1 is the year
            level_1 = self.result_dataframe.index.get_level_values(1).unique()    
            for year in level_1:
                file = filename + '-' + str(year)[0:4] + '.xlsx'
                df_r = self.result_dataframe.xs(year, axis = 0, level = 1, drop_level = False)
                if len(df_r.index) > 0:
                    to_excel(structs  = self.metapatterns, 
                             df_patterns = self.pattern_dataframe,
                             df_patterns_short = self.pattern_dataframe_short,
                             df_results  = df_r, 
                             path     = path, 
                             filename = file)
                if year == level_1[-1]: # if last year then create files per entity
                    name_list = self.result_dataframe.index.get_level_values(0).unique()    
                    for name in name_list:
                        file = filename + '-' + str(year)[0:4] + "-" + name[name.find("(")+1:name.find(")")] + '.xlsx'
                        df_r = self.result_dataframe.xs(year, axis = 0, level = 1, drop_level = False).xs(name, axis = 0, level = 0, drop_level = False)
                        if len(df_r.index) > 0: # only write is they already sent in reports
                            to_excel(structs  = self.metapatterns, 
                                     df_patterns = self.pattern_dataframe, 
                                     df_patterns_short = self.pattern_dataframe_short,
                                     df_results  = df_r, 
                                     path     = path, 
                                     filename = file)

        else:
            to_excel(structs = self.metapatterns, 
                     df_patterns = self.pattern_dataframe, 
                     df_patterns_short = self.pattern_dataframe_short,
                     df_results  = self.result_dataframe, 
                     path     = path, 
                     filename = filename + ".xlsx")          
        return None
