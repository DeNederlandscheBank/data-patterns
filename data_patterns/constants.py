'''Encoding definitions
'''

__author__ = """De Nederlandsche Bank"""
__email__ = 'ECDB_berichten@dnb.nl'
__version__ = '0.1.6'

PATTERN_ID       	   		= "pattern_id"
CLUSTER          	   		= "cluster"
P_COLUMNS        	   		= "P columns"
RELATION_TYPE    	   		= "relation type"
Q_COLUMNS        	   		= "Q columns"
P_PART           	   		= "P"
RELATION         	   		= "relation"
Q_PART           	   		= "Q"
SUPPORT          	   		= "support"
EXCEPTIONS       	   		= "exceptions"
CONFIDENCE       	   		= "confidence"
PATTERN_STATUS   	   		= "pattern status"
ENCODINGS        	   		= "encodings"
PANDAS_CO					= "pandas co"
PANDAS_EX					= "pandas ex"

RESULT_TYPE      	   		= 'result_type'
INDEX            	   		= "index"
P_VALUES         	   		= "P values"
Q_VALUES         	   		= "Q values"

ENCODE            	   		= "encode"

INITIAL_PATTERN_STATUS 		= 'not defined'
TEXT_CONFIRMATION	   		= 'confirmation'
TEXT_EXCEPTION		   		= 'exception'

DEFAULT_SHEET_NAME_PATTERNS = 'Patterns'
SHEET_NAME_POST_CO     		= "_co"
SHEET_NAME_POST_EX     		= "_ex"

PATTERNS_COLUMNS 	   		= [PATTERN_ID, CLUSTER, 
					  		   P_COLUMNS, RELATION_TYPE, Q_COLUMNS, P_PART, RELATION, Q_PART, 
                    	  	   SUPPORT, EXCEPTIONS, CONFIDENCE, PATTERN_STATUS, ENCODINGS, PANDAS_CO, PANDAS_EX]
     
RESULTS_COLUMNS        		= [RESULT_TYPE, PATTERN_ID, CLUSTER, INDEX, SUPPORT, EXCEPTIONS, CONFIDENCE, 
                    	  	   P_COLUMNS, RELATION_TYPE, Q_COLUMNS, P_PART, RELATION, Q_PART, P_VALUES, Q_VALUES]

