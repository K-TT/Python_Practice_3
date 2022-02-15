Python 3.9.7 (v3.9.7:1016ef3790, Aug 30 2021, 16:39:15) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandes as pd
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    import pandes as pd
ModuleNotFoundError: No module named 'pandes'
>>> import pandas as pd
>>> series=pd.Series([1,2,3,4,5])
>>> series
0    1
1    2
2    3
3    4
4    5
dtype: int64
>>> series2=pd.Series([1,2,3,4,5]), idex['a','b','c','d','e'])
SyntaxError: unmatched ')'
>>>  series2=pd.Series([1,2,3,4,5], idex['a','b','c','d','e'])
 
SyntaxError: unexpected indent
>>> series2=pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
>>> series2
a    1
b    2
c    3
d    4
e    5
dtype: int64
>>> series[3]
4
>>> series2['d']
4
>>> series.iloc[3]
4
>>> series2.iloc[2]
3
>>> dates1=pd.date_range('20210719',periods=12)
>>> dates2=pd.date_range('20210719',periods=12,freq='Y')
>>> dates3=pd.date_range('20210719',periods=12,freq='M')
>>> series3=pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
>>> series3.index=dates3
>>> series3
2021-07-31     1
2021-08-31     2
2021-09-30     3
2021-10-31     4
2021-11-30     5
2021-12-31     6
2022-01-31     7
2022-02-28     8
2022-03-31     9
2022-04-30    10
2022-05-31    11
2022-06-30    12
Freq: M, dtype: int64
>>> pd.read_csv(r'd:\test.csv')
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    pd.read_csv(r'd:\test.csv')
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 586, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 482, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 811, in __init__
    self._engine = self._make_engine(self.engine)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1040, in _make_engine
    return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 51, in __init__
    self._open_handles(src, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/base_parser.py", line 222, in _open_handles
    self.handles = get_handle(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/common.py", line 702, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:\\test.csv'
>>> dir(iris.csv)
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    dir(iris.csv)
NameError: name 'iris' is not defined
>>> iris.csv
Traceback (most recent call last):
  File "<pyshell#20>", line 1, in <module>
    iris.csv
NameError: name 'iris' is not defined
>>> pd.read_csv(r'd:\test.csv')
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    pd.read_csv(r'd:\test.csv')
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 586, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 482, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 811, in __init__
    self._engine = self._make_engine(self.engine)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1040, in _make_engine
    return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 51, in __init__
    self._open_handles(src, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/base_parser.py", line 222, in _open_handles
    self.handles = get_handle(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/common.py", line 702, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:\\test.csv'
>>> pd.read_csv(r'd:\test.csv')
Traceback (most recent call last):
  File "<pyshell#22>", line 1, in <module>
    pd.read_csv(r'd:\test.csv')
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 586, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 482, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 811, in __init__
    self._engine = self._make_engine(self.engine)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1040, in _make_engine
    return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 51, in __init__
    self._open_handles(src, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/base_parser.py", line 222, in _open_handles
    self.handles = get_handle(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/common.py", line 702, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:\\test.csv'
>>> pd.read_csv(r'd:Library\test.csv')
Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    pd.read_csv(r'd:Library\test.csv')
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/util/_decorators.py", line 311, in wrapper
    return func(*args, **kwargs)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 586, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 482, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 811, in __init__
    self._engine = self._make_engine(self.engine)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/readers.py", line 1040, in _make_engine
    return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 51, in __init__
    self._open_handles(src, kwds)
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/parsers/base_parser.py", line 222, in _open_handles
    self.handles = get_handle(
  File "/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/pandas/io/common.py", line 702, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'd:Library\\test.csv'
>>> pd.read_csv('/Users/kateryna/Desktop/test.csv')
     150    4  setosa  versicolor  virginica
0    5.1  3.5     1.4         0.2          0
1    4.9  3.0     1.4         0.2          0
2    4.7  3.2     1.3         0.2          0
3    4.6  3.1     1.5         0.2          0
4    5.0  3.6     1.4         0.2          0
..   ...  ...     ...         ...        ...
145  6.7  3.0     5.2         2.3          2
146  6.3  2.5     5.0         1.9          2
147  6.5  3.0     5.2         2.0          2
148  6.2  3.4     5.4         2.3          2
149  5.9  3.0     5.1         1.8          2

[150 rows x 5 columns]
>>> pd.read_csv('/Users/kateryna/Desktop/test3.txt')
        A B C
0     0 1 2 3
1     1 4 5 6
2     2 7 8 9
3  3 11 11 12
>>> df_test=pd.read_csv('/Users/kateryna/Desktop/test.csv')
>>> df_test.describe()
              150           4      setosa  versicolor   virginica
count  150.000000  150.000000  150.000000  150.000000  150.000000
mean     5.843333    3.057333    3.758000    1.199333    1.000000
std      0.828066    0.435866    1.765298    0.762238    0.819232
min      4.300000    2.000000    1.000000    0.100000    0.000000
25%      5.100000    2.800000    1.600000    0.300000    0.000000
50%      5.800000    3.000000    4.350000    1.300000    1.000000
75%      6.400000    3.300000    5.100000    1.800000    2.000000
max      7.900000    4.400000    6.900000    2.500000    2.000000
>>> df1000=pd.DataFrame(np.random.randn(1000,4),columns=list('ABCD'))
Traceback (most recent call last):
  File "<pyshell#28>", line 1, in <module>
    df1000=pd.DataFrame(np.random.randn(1000,4),columns=list('ABCD'))
NameError: name 'np' is not defined
>>> import numpy as np
>>> df1000=pd.DataFrame(np.random.randn(1000,4),columns=list('ABCD'))
>>> df1000.describe()
                 A            B            C            D
count  1000.000000  1000.000000  1000.000000  1000.000000
mean      0.003987    -0.021660    -0.002108    -0.034201
std       0.995159     1.016476     0.999656     1.021265
min      -3.207165    -2.821828    -3.095667    -3.525137
25%      -0.668133    -0.746134    -0.677145    -0.752548
50%      -0.031289     0.007448     0.012479    -0.034563
75%       0.688614     0.672829     0.703070     0.684239
max       3.311012     3.432125     3.059012     3.008277
>>> df1000.mean()
A    0.003987
B   -0.021660
C   -0.002108
D   -0.034201
dtype: float64
>>> 