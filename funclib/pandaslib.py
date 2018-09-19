# pylint: disable=C0302, too-many-branches, dangerous-default-value, line-too-long, no-member, expression-not-assigned, locally-disabled,not-context-manager, unused-import, undefined-variable, no-member
'''routines to manipulate array like objects like lists, tuples etc'''
from warnings import warn
import os.path as _path
import math as _math

import statsmodels.stats.api as _sms
import pandas as _pd
import numpy as _np
from scipy import stats as _stats
from sklearn.metrics import mean_squared_error as _mean_squared_error

from funclib.baselib import list_flatten as _list_flatten
from funclib.iolib import PrintProgress as _PrintProgress
import funclib.iolib as _iolib
from funclib.to_precision import std_notation as _std_notation

# region Pandas

class GroupBy():
    '''Wrapper for the Pandas group_by function.

    Includes a library of aggregate functions which allow arguments
    to be passed to the function, allowing calculation of spefic metrics
    e.g. a custom percentile.

    GroupBy.PRECISION controls number formats for some functions

    Arguments:
        df: pandas dataframe
        groupflds: iterable list of fields to aggregate by
        valuefls: list of value fields with the data to summarise
        *funcs: list of aggregate functions

    funcs can be custom static functions (e.g. static members
    of this class, or other scipy or numpy functions)


    Example:
    >>>import pandaslib.GroupBy as GroupBy
    >>>import numpy as np
    >>>df = <LOAD A DATA FRAME>
    >>>GroupBy.PRECISION = 1
    >>>GB = GroupBy(df, ['fish', 'sex'], ['length', 'weight'], np.mean, np.median, GroupBy.fCI(95), GroupBy.fPercentile(25))
    >>>GB.to_excel('C:\temp\tmp.xlsx', fail_if_exists=False)
    '''

    PRECISION = 2

    def __init__(self, df, groupflds, valueflds, *funcs):
        self.df = df
        self._groupflds = groupflds
        self._valueflds = valueflds
        self._funcs = funcs
        self.result = None
        self.execute()


    def execute(self):
        '''()->DataFrame
        '''
        #build dict
        allfuncs = self._funcs * len(self._valueflds)
        d = {}
        for v in self._valueflds:
            d[v] = self._funcs

        self.result = self.df.groupby(self._groupflds).agg(d)
        return self.result


    def to_excel(self, fname, sheetname='Sheet1', fail_if_exists=True, openfld=True):
        '''save as excel'''
        if not isinstance(self.result, _pd.DataFrame):
            warn('Aggregate results do not exist')
            return
        fname = _path.normpath(fname)
        fld, f, _ = _iolib.get_file_parts2(fname)
        if _iolib.file_exists(fname) and fail_if_exists:
            raise FileExistsError('File %s exists and fail_if_exists==True' % f)

        assert isinstance(self.result, _pd.DataFrame)
        self.result.to_excel(fname, sheetname)
        if openfld:
            try:
                _iolib.folder_open(fld)
            except:
                pass

    @staticmethod
    def fPercentile(n):
        def percentile_(data):
            return _np.percentile(data, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    @staticmethod
    def fMSE(pred):
        '''Mean squared error
        Assess quality of estimator
        MSE = 1/n * sum(x - xmodel)^2

        pred is the expected value, i.e. model mean
        '''
        def mse_(data):
            ndpred = _np.zeros(data.shape[0]) + pred
            return _mean_squared_error(ndpred, data)
        mse_.__name__ = 'mse_%s' % pred
        return mse_

    def fRMSE(pred):
        '''Root mean squared error.
        Assess quality of estimator.
        RMSE is measured on same scale and units as the dependent variable.
        RMSE =  sqrt( 1/n * sum(x - xmodel)^2 )
        pred is the expected value, i.e. model mean
        '''
        def rmse_(data):
            ndpred = _np.zeros(data.shape[0]) + pred
            x = _mean_squared_error(ndpred, data)**0.5
            return  x
        rmse_.__name__ = 'rmse_%s' % pred
        return rmse_

    @staticmethod
    def fCI(interval=95):
        def ci_(data):
            a = _sms.DescrStatsW(data).tconfint_mean((100 - interval)*0.01)
            return (max(a) - min(a)) / 2.
        ci_.__name__ = 'ci_%s' % interval
        return ci_

    @staticmethod
    def fCILower(interval=95):
        def CILower_(data):
            l, u = _sms.DescrStatsW(data).tconfint_mean((100 - interval)*0.01)
            return l
        CILower_.__name__ = 'CILower_%s' % interval
        return CILower_

    @staticmethod
    def CIUpper(interval=95):
        def CIUpper_(data):
            l, u = _sms.DescrStatsW(data).tconfint_mean((100 - interval)*0.01)
            return l
        CIUpper_.__name__ = 'CIUpper_%s' % interval
        return CIUpper_

    @staticmethod
    def fCI_str(interval=95):
        '''formatted str version'''
        def CI_str_(data):
            l, u = _sms.DescrStatsW(data).tconfint_mean((100 - interval)*0.01)
            m = _np.mean(data)
            s = 'M=%s %s%% CIs [%s, %s]' % (_std_notation(m, GroupBy.PRECISION), interval, _std_notation(l, GroupBy.PRECISION), _std_notation(u, GroupBy.PRECISION))
            #return m, m-h, m+h
            return s
        CI_str_.__name__ = 'CI_str_%s' % interval
        return CI_str_

    @staticmethod
    def n(data):
        d = _np.array(data)
        return _np.count_nonzero(d)

    @staticmethod
    def fMeanSD_str(data):
        d = _np.array(data)
        m, sd = _np.mean(a), _np.std(a)
        s = '%s ±%s' % (_std_notation(m, GroupBy.PRECISION), _std_notation(sd, GroupBy.PRECISION))
        return s



def pd_df_to_ndarray(df):
    '''(dataframe)->ndarray
    Return a dataframe as a numpy array
    '''
    return df.as_matrix([x for x in df.columns])


def col_append(df, col_name):
    '''(df,str)->df
    df is BYREF
    adds a column to dataframe filling it
    with np.NaN values.
    '''
    df.loc[:, col_name] = _pd.Series(_pd.np.nan, index=df.index)


def col_append_nan_fill(df, col_name):
    '''(df,str)->df
    df is BYREF
    adds a column to dataframe filling it
    with np.NaN values.
    '''
    col_append(df, col_name)


def col_append_fill(df, col_name, f):
    '''(df,str,any)->df
    df is BYREF
    adds a column to dataframe filling it with value f
    If f is None, filled with NaN
    '''
    if f is None:
        df.loc[:, col_name] = _pd.Series(_pd.np.nan, index=df.index)
    else:
        df.loc[:, col_name] = _pd.Series(f, index=df.index)


def col_append_rand_fill(df, col_name, lower=0, upper=1):
    '''(df,str,any)->df
    df is BYREF
    adds a column to dataframe filling it with random values from a standard normal
    '''
    df[col_name] = _np.random.choice(range(lower, upper), df.shape[0])


def col_calculate_new(df, func, new_col_name, *args, progress_init_msg='\n'):
    '''(pd.df, function, str, the named arguments for function)
    1) Adds a new column called col_name
    2) calls func with args by position,  where args are the row indices for the values
    3) Row indexes are ZERO based
    4) Consider using apply or similiar for simple use cases

    df = pd.dataframe({'a':[1,2],'b'=[10,20]})

    DF=
    a   b
    1   10
    2   20

    def f(a, b):
        return a*b

    func = f
    col_calculate_new(df, func, 'product', 0, 1)

    DF=
    a   b   product
    1   10  10
    2   20  40
    '''
    assert isinstance(df, _pd.DataFrame)
    if new_col_name in df.columns:
        raise BaseException('Column %s already exists in the dataframe.' % (new_col_name))
    col_append(df, new_col_name)

    args = list(args)
    args = _list_flatten(args)
    PP = _PrintProgress(len(df.index), init_msg=progress_init_msg)
    for i, row in df.iterrows():
        PP.increment()
        rowvals = []
        for x in args:
            if row[x] is None:
                vv = None
            elif _np.isnan(row[x]):
                vv = None
            else:
                vv = row[x]
            rowvals.append(vv)
        v = func(*rowvals)
        df.set_value(i, new_col_name, v)


def col_exists(df, col_name):
    '''(str)->bool
    '''
    return col_name in df.columns


def col_index(df, col_name):
    '''(df, str)->int
    Given col return index
    Returns None if doesnt exist
    '''
    if col_exists(df, col_name):
        return df.columns.get_loc(col_name)
    return None


def cols_get_indexes_from_names(df, *args):
    '''df, str args->tuple
    Given list if strings get the corresponding
    column indexes and return as a tuple
    '''
    return [col_index(df, x) for x in args]


def readfld(v, default=None):
    '''return default if v is a pandas null
    '''
    return default if _pd.isnull(v) else v
# endregion


def df_mssql(sql, dbname, server='(local)', port=1433, security='integrated', user='', pw=''):
    '''(str, str, str, str, str)-> pandas.dataframe
    Get a pandas dataframe from SQL Server

    sql: the sql to execute to get the data
    dbname: database name
    server:server identifier
    security: integrated or sqlserver
    user,pw: SQL server user authentication details
    '''
