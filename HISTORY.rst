=======
History
=======

0.1.0 (2019-10-27)
------------------

* Development release.

0.1.11 (2019-11-6)
------------------

* First release on PyPI.


< 0.1.17 (2020-10-6)
---------------------
    
    Expression
    
You can now use expressions to find patterns. This is a string such as '{.*}={.*}' (this one will find columns that are equal to eachother). See example in usage as how to do it, also with unknown values. 

Patterns of the for IF THEN will be done through a pandas expression and quantitative patterns will be found using numpy (quicker). Expression will be split up in parts if it is quantitative

    Function
   
Added the function correct_data. This corrects data based on the most common value if grouped with another column, e.g. changes the names in a column if there are multiple names per LEI code. 


    Other
    
1. Added P and Q values to analyze

2. highest_conf option to find the pattern with the highest conf based on P value.

3. Possible to use with EVA2 rules


0.1.17 (2020-10-6)
------------------

    Parameters
    
1. 'window' (boolean): Only compares columns in a window of n, so [column-n, column+n].

2. 'disable' (boolean): If you set this to True, it will disable all tqdm progress bars for finding and analyzing patterns.

3. 'expres' (boolean): If you use an expression, it will only directly work with the expression if it is an IF THEN statement. Otherwise it is a quantitative pattern and it will be split up in parts and it uses numpy to find the patterns (this is quicker). However sometimes you want to work with an expression directly, such as the difference between two columns is lower than 5%. If you set expres to True, it will work directly with the expression. 



    Expression

1. You can use ABS in expressions. This calculates the absolute value. So something like 'ABS({'X'} - {'Y'}) = {'Z'})'



    cluster
    
1. You can now add the column name on which you want to cluster


    Function
    
1. Convert_to_time: merge periodes together by adding suffix to columns (t-1) and (t).

2. convert_columns_to_time: Make the periods into columns so that you have years as columns.


    Other
    
1. Add tqdm progress bars 


0.1.18 (16-11-2020)
--------------------

    variables to miner
    
You can now add a boolean to the miner. If you give the boolean True to the miner, it will get rid of all the " and ' in the string data. This is needed for some data where name have those characters in their name. This will give errors later on if not removed.



    Function to read overzicht
    
    
    Changed the IF THEN expression so that we can use decimals when numeric
    
       
    Parameters
    
1. 'notNaN' (boolean): Only takes not NaN columns

    Function changes
    
1. Convert_to_time: add boolean set_year. If true then only use the years (this is for yearly data), otherwise keep whole date. Set to True standard

2. update_statistics: Remove patterns that contain columns which are not in the data. This is necessary for some insurers so that they do not get errors

