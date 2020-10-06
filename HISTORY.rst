=======
History
=======

0.1.0 (2019-10-27)
------------------

* Development release.

0.1.11 (2019-11-6)
------------------

* First release on PyPI.

0.1.17 (2020-10-6)
--------------

    Parameters
    
1. 'window' (boolean): Only compares columns in a window of n, so [column-n, column+n].

2. 'disable' (boolean): If you set this to True, it will disable all tqdm progress bars for finding and analyzing patterns.

3. 'expres' (boolean): If you use an expression, it will only directly work with the expression if it is an IF THEN statement. Otherwise it is a quantitative pattern and it will be split up in parts and it uses numpy to find the patterns (this is quicker). However sometimes you want to work with an expression directly, such as the difference between two columns is lower than 5%. If you set expres to True, it will work directly with the expression. 



    Expression

1. You can use ABS in expressions. This calculates the absolute value. So something like 'ABS({'X'} - {'Y'}) = {'Z'})'



    cluster
    
1. You can now add the column name on which you want to cluster
