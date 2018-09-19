# pylint: disable=too-few-public-methods, too-many-statements, unused-import, unused-variable, no-member, dangerous-default-value, unbalanced-tuple-unpacking
'''My library of statistics based functions'''

# base imports
import copy
import numbers
import warnings

# packages
import pandas as _pd


import numpy as _np
import scipy as _scipy
import scipy.stats as _stats
import statsmodels.api as sm


# mine
from enum import Enum
from funclib.baselib import switch
import funclib.baselib as _baselib
import funclib.iolib as _iolib
import funclib.arraylib as _arraylib


# region Enuerations
class EnumMethod(Enum):
    '''enum used to select the type of correlation analysis'''
    kendall = 1
    spearman = 2
    pearson = 3


# region Correlations
def permuted_correlation(list_a, list_b, test_stat, iterations=0, method=EnumMethod.kendall, out_greater_than_test_stat=0):
    '''(list, list, float, int, enumeration, list[byref], list[byref], enumeration) -> list
    Return a list containing multiple [n=iterations] of correlation test statistic and p values
    following permutation. e_method is an enumeration member of e_method.

    test_stat is the actual x,y calulated correlation value

    list_a and list_b should be passed in their original (paired) order
    for the exclude zero pairs to work.

    Iterations is manual number of iterations (ignores max_iterations).
    '''
    assert isinstance(list_a, list), 'list_a should be a list'
    assert isinstance(list_b, list), 'list_b should be a list'
    assert isinstance(
        iterations, numbers.Number), 'iterations should be an int'

    if iterations == 0:
        raise ValueError('Iterations cannot be zero')

    results = []
    justtest = []
    # we will permute list_b but don't need to permute list_a
    permuted = copy.deepcopy(list_b)
    cnt = 0
    for dummy in range(0, int(iterations)):
        cnt += 1
        permuted = _np.random.permutation(permuted)

        for case in switch(method):
            if case(EnumMethod.kendall):
                teststat, pval = _stats.kendalltau(list_a, permuted)
                break
            if case(EnumMethod.pearson):
                teststat, pval = _stats.pearsonr(list_a, permuted)
                break
            if case(EnumMethod.spearman):
                teststat, pval = _stats.spearmanr(list_a, permuted)
                break
            if case():
                raise ValueError('Enumeration member not in e_method')

        if teststat > test_stat:
            out_greater_than_test_stat[0] += 1

        results.append([teststat, pval])
        justtest.append([teststat])
        pre = '/* iter:' + str(cnt) + ' */'
        _iolib.print_progress(cnt, iterations, prefix=pre, bar_length=30)

    return results


def correlation_test_from_csv(file_name_or_dataframe, col_a_name, col_b_name, test_type=EnumMethod.kendall):
    ''' (string|_pd.dataframe,string,string,EnumMethod,EnumStatsEngine) -> dictionary
    Assumes that the first row is headers, will fail if this is not the case.

    NOTE:This opens file_name each time so dont recommend it for repeating tests.

    Returns:
        {'teststat':, 'p':, 'n':}

    '''
    if isinstance(file_name_or_dataframe, str):
        df = _pd.read_csv(file_name_or_dataframe)
    else:
        df = file_name_or_dataframe
    assert isinstance(df, _pd.DataFrame)

    list_a = df[col_a_name].tolist()
    list_b = df[col_b_name].tolist()

    for case in _baselib.switch(test_type):
        if case(EnumMethod.kendall):
            teststat, pval = _stats.kendalltau(list_a, list_b)
            break
        if case(EnumMethod.pearson):
            teststat, pval = _stats.pearsonr(list_a, list_b)
            break
        if case(EnumMethod.spearman):
            teststat, pval = _stats.spearmanr(list_a, list_b)
            break
        if case():
            raise ValueError('Enumeration member not in e_method')

    return {'teststat': teststat, 'p': pval, 'n': len(list_a)}


def correlation(a, b, method=EnumMethod.kendall):
    '''(list|ndarray, list|ndarray, enumeration, enumeration) -> dict
    Returns a dictionary: {'teststat':teststat, 'p':pval}
    method: is an enumeration member of EnumMethod
    engine: is an enumeration method
    scipy cant cope with nans. Matched nans will be removed if a and b are numpy arrays
    '''

    if isinstance(a, _np.ndarray) or isinstance(b, _np.ndarray):
        if isinstance(a, _np.ndarray) is False or isinstance(b, _np.ndarray) is False:
            raise ValueError(
                'If numpy arrays are used, both must be ndarray types')

        if a.shape != b.shape:
            raise ValueError('Numpy array shapes must match exactly')

        # scipy doesnt like nans. We drop out paired nans, leaving
        # all other pairings the same
        if _arraylib.np_contains_nan(a) and _arraylib.np_contains_nan(b):
            dic = _arraylib.np_delete_paired_nans_flattened(a, b)
        else:
            dic = {'a': a, 'b': b}

        # we have unmatched nans, ie a nan in one array
        # with a scalar in the other
        # this is an error state - could modify later to exclude
        # all values from both arrays where there is any nan
        if _arraylib.np_contains_nan(dic['a']):
            raise ValueError('Numpy array a contains NaNs')

        if _arraylib.np_contains_nan(dic['b']):
            raise ValueError('Numpy array b contains NaNs')

        lst_a = dic['a'].flatten().tolist()
        lst_b = dic['b'].flatten().tolist()
    else:
        if isinstance(a, list) is False or isinstance(b, list) is False:
            raise ValueError('If lists are used, both must be list types')
        lst_a = copy.deepcopy(a)
        lst_b = copy.deepcopy(b)

    if len(lst_a) != len(lst_b):
        raise ValueError('Array lengths must match exactly')

    assert isinstance(lst_a, list)
    assert isinstance(lst_b, list)

    for case in _baselib.switch(method):
        if case(EnumMethod.kendall):
            teststat, pval = _stats.kendalltau(lst_a, lst_b)
            break
        if case(EnumMethod.pearson):
            teststat, pval = _stats.pearsonr(lst_a, lst_b)
            break
        if case(EnumMethod.spearman):
            #if engine == EnumStatsEngine.r:
            #    df = _pd.DataFrame({'a': lst_a, 'b': lst_b})
            #    df_r = _rpy2.robjects.pandas2ri(df)
            #    _ro.globalenv['cordf'] = df_r
            #    tmpstr = 'cor.test(cordf$a, cordf$b, method="spearman")'
            #    result = _ro.r(tmpstr)
            #    teststat = result[3][0]
            #    pval = result[2][0]
            #else:
            teststat, pval = _stats.spearmanr(lst_a, lst_b)
            break
        if case():
            raise ValueError('Enumeration member not in e_method')

    return {'teststat': teststat, 'p': pval}


def permuted_teststat_check(csvfile, test_col, stat_value_to_check):
    '''(str, int, int, float) -> dic ({'p', 'n', 'more_extreme_n', 'teststat'})
    csvfile: full data file file name
    start_row: row with first line of data (0 based!), pass None if there is are no headers
    test_col: column with test stats, note this is a 0 based index, first column would be indexed as 0.
    stat_value_to_check: the unpermuted test stat result

    Returns a dictionary object:
    'p':   probability of getting this result by chance alone
    'n':   number of values checked
    'more_extreme_n': number of values more extreme than stat_value_to_check
    '''
    col = [test_col]
    df = _pd.read_csv(csvfile, usecols=col)
    assert isinstance(df, _pd.DataFrame)

    if stat_value_to_check >= 0:
        res = (df.iloc[0:] > stat_value_to_check).sum()
    else:
        res = (df.iloc[0:] < stat_value_to_check).sum()

    return {'p': float(res[0]) / (df.iloc[0:]).count()[0], 'n': (df.iloc[0:]).count()[0], 'more_extreme_n': res[0]}


def permuted_teststat_check1(teststats, stat_value_to_check):
    '''(tuple|list) -> dic ({'p', 'n', 'more_extreme_n'})
    teststats is an iterable
    stat_value_to_check: the unpermuted test stat result

    Returns a dictionary object:
    'p':   probability of getting this result by chance alone
    'n':   number of values checked
    'more_extreme_n': number of values more extreme than stat_value_to_check
    '''
    v = 0
    p = 0
    if stat_value_to_check >= 0:
        for v in teststats:
            if v > stat_value_to_check:
                p += 1
    else:
        for v in teststats:
            if v < stat_value_to_check:
                p += 1

    return {'p': float(p) / len(teststats), 'n': len(teststats), 'more_extreme_n': p}


def focal_permutation(x, y, teststat, iters=1000):
    '''(ndarray, ndarray) -> dic
    Perform 3x3 mean and then a permutation correlation test on two 2D arrays
    Calculates p
    dic  {'p':,'more_extreme_n':}
    '''

    taus = []
    # only need to do focal mean once on y as this doesnt need to be permuted
    # each time
    y = _arraylib.np_focal_mean(y, False)

    for cnt in range(iters):
        pre = '/* iter:' + str(cnt + 1) + ' */'

        # just permute one of them
        a = _arraylib.np_permute_2d(x)
        a = _arraylib.np_focal_mean(a, False)

        # use scipy - p will be wrong, but the taus will be right
        res = correlation(a, y)
        taus.append(res['teststat'])

        _iolib.print_progress(cnt + 1, iters, prefix=pre, bar_length=30)

    dic = permuted_teststat_check1(taus, teststat)
    return {'p': dic['p'], 'more_extreme_n': dic['more_extreme_n']}
# endregion


# region Binning and recoding
def quantile_bin(nd, percentiles=None, zero_as_zero=False):
    '''(ndarray, list, list, boolean) -> ndarray

    percentiles is list of percentiles, defaults to [25, 50, 100]

    Returned array has elements with labels substituted.
    Array
    of floats, nans ignored in calculations and returned as nans
    '''

    def get_quantile_ranges(nd, percentiles, exclude_zeros=False, use_scipy=True):
        '''(ndarray like, listlike, bool) -> list
        get the full intervals
        percentiles=[25,50,75] would give quartiles

        Returns array of arrays size percentiles+1 x 3.
        [[0,1.1,bins[0]
          1.1,2.2,bins[1]]
          where the first two figures in the inner list of the numerical ranges
          for the data in array nd. The last figure is a number which represents the percentile
          to which the range belongs (starts at 1).
          For example: percentiles=[25,50,75]
          0%-25%=1 25%-50%=2 50%-75%=3 75%-100%=4

          If zeros are excluded they are assigned as 0.


        if exclue_zeros is true, then zeros will be excluded from quantile calculations
        '''
        assert isinstance(percentiles, list)
        #assert isinstance(nd, numpy.ndarray)

        ret = []
        a = _np.array(nd).flatten()  # default dtype is float
        assert isinstance(a, _np.ndarray)

        if exclude_zeros:
            _np.place(a, a == 0, _np.nan)

        a = _arraylib.np_delete_zeros(a)
        labels = range(1, len(percentiles) + 2)  # [25,50,75] -> [1,2,3,4]

        percentiles.sort()
        ranges = [_stats.scoreatpercentile(
            a, x) for x in percentiles] if use_scipy else _np.percentile(a, percentiles)

        for ind, item in enumerate(ranges):
            if ind == 0:
                ret.append([a.min() - 0.00001, item, labels[ind]])
            else:
                ret.append([ranges[ind - 1], item, labels[ind]])

        # add last category
        ret.append([ranges[-1], a.max() + 0.00001, len(labels)])

        return ret

    def get_bin_label(x, ranges, zero_as_zero=False):
        '''(numeric, list, bool) -> string
        Uses the structure return by _get_quantile_ranges

        Zero as zero forces zeros to be excluded from all
        quantile calculations and put under their own label of 0
        '''
        assigned = False
        for ind, item in enumerate(ranges):
            if x == 0 and zero_as_zero:
                ret = 0
                assigned = True
                break
            elif _np.isnan(x):
                ret = _np.nan
                assigned = True
                break
            elif item[0] < x <= item[1]:
                ret = item[2]  # this is the label index eg 'High'
                assigned = True
                break

        if not assigned:
            raise ValueError('Failed to assign bin label')
        return ret

    if percentiles is None:
        percentiles = [25, 50, 100]
    if nd.dtype != float:
        raise ValueError(
            'Array should be of type float. Try recasting using ndarray.astype(float)')

    gqr = get_quantile_ranges(nd, percentiles, zero_as_zero)
    out = _np.copy(nd)
    func = _np.vectorize(get_bin_label, excluded=['ranges', 'zero_as_zero'])
    out = func(out, ranges=gqr, zero_as_zero=zero_as_zero)

    return out
# endregion


# region Contingency
def contingency_conditional(a, bycol=True):
    '''(ndarray, bool)->ndarray
    calculates conditional contingency by rows or columns
    as specified by bycol

    Also adds a marginal row (bycol=False) or marginal col (bycol=True)
    '''
    assert isinstance(a, _np.ndarray)
    b = contigency_joint(a)
    assert isinstance(b, _np.ndarray)
    marg_rows, marg_cols = _stats.contingency.margins(b)

    if bycol:
        b = _np.vstack([b, marg_cols])  # add marginal col
        # loop through each col and use the column marginal to calculate
        # conditional
        for i in range(int(b.shape[1])):
            b[0:-1, i:i + 1] = b[0:-1, i:i + 1] / b[-1, i:i + 1]
    else:
        b = _np.hstack([b, marg_rows])
        for i in range(int(b.shape[0])):
            b[i:i + 1, 0:-1] = b[i:i + 1, 0:-1] / b[i:i + 1, -1]
    return b


def contigency_joint(a):
    '''(ndarray, bool)->ndarray
    calculates conditional contingency by rows or columns
    as specified by bycol
    '''
    assert isinstance(a, _np.ndarray)
    b = a.astype(float)
    return b / _np.sum(b)
# endregion


def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = _np.histogram(data, bins=bins, density=True)
    x = (x + _np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        _stats.alpha, _stats.anglit, _stats.arcsine, _stats.beta, _stats.betaprime, _stats.bradford, _stats.burr, _stats.cauchy, _stats.chi, _stats.chi2, _stats.cosine,
        _stats.dgamma, _stats.dweibull, _stats.erlang, _stats.expon, _stats.exponnorm, _stats.exponweib, _stats.exponpow, _stats.f, _stats.fatiguelife, _stats.fisk,
        _stats.foldcauchy, _stats.foldnorm, _stats.frechet_r, _stats.frechet_l, _stats.genlogistic, _stats.genpareto, _stats.gennorm, _stats.genexpon,
        _stats.genextreme, _stats.gausshyper, _stats.gamma, _stats.gengamma, _stats.genhalflogistic, _stats.gilbrat, _stats.gompertz, _stats.gumbel_r,
        _stats.gumbel_l, _stats.halfcauchy, _stats.halflogistic, _stats.halfnorm, _stats.halfgennorm, _stats.hypsecant, _stats.invgamma, _stats.invgauss,
        _stats.invweibull, _stats.johnsonsb, _stats.johnsonsu, _stats.ksone, _stats.kstwobign, _stats.laplace, _stats.levy, _stats.levy_l, _stats.levy_stable,
        _stats.logistic, _stats.loggamma, _stats.loglaplace, _stats.lognorm, _stats.lomax, _stats.maxwell, _stats.mielke, _stats.nakagami, _stats.ncx2, _stats.ncf,
        _stats.nct, _stats.norm, _stats.pareto, _stats.pearson3, _stats.powerlaw, _stats.powerlognorm, _stats.powernorm, _stats.rdist, _stats.reciprocal,
        _stats.rayleigh, _stats.rice, _stats.recipinvgauss, _stats.semicircular, _stats.t, _stats.triang, _stats.truncexpon, _stats.truncnorm, _stats.tukeylambda,
        _stats.uniform, _stats.vonmises, _stats.vonmises_line, _stats.wald, _stats.weibull_min, _stats.weibull_max, _stats.wrapcauchy
    ]

    # Best holders
    best_distribution = _stats.norm
    best_params = (0.0, 1.0)
    best_sse = _np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = _np.sum(_np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        _pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Propbability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into _pd Series
    x = _np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = _pd.Series(y, x)

    return pdf


def linreg(X, Y):
    '''(ndarray|list, ndarray|list -> statsmodel.model)

    Pass in two 1D numpy arrays or lists, returns an
    instance of statsmodel.model

    X, Y: Independent and independent 1d array of values

    Example
    >>>X = np.arange(10)
    >>>Y = X*2
    >>>model = linreg(X, Y)
    '''
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    return model


def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5




def _mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = _mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss