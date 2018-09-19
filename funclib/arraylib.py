# pylint: disable=C0302, dangerous-default-value, no-member, expression-not-assigned, not-context-manager, invalid-name
'''routines to manipulate array like objects like lists, tuples etc'''
from warnings import warn as _warn

import numpy as _np
import numpy.random
import scipy.ndimage as _ndimage
from scipy.spatial import distance as dist


from funclib.baselib import list_flatten as _list_flatten

# region NUMPY
def  vstackt(arrays):
    '''(list:ndarray) - > ndarray
    vstack arrays, cropping to smallest width

    arrays:
        list of ndarrays

    Returns:
        single ndarray, a vstack of arrays
    '''
    widths = [a.shape[1] for a in arrays]

    for i, a in enumerate(arrays):
        if i == 0:
            out = a[:, 0:min(widths), :]
        else:
            out = _np.vstack([out, a[:, 0:min(widths), :]])
    return out


def  hstackt(arrays):
    '''(list:ndarray) - > ndarray
    hstack arrays, cropping to smallest height

    arrays:
        list of ndarrays

    Returns:
        single ndarray, a hstack of arrays
    '''
    heights = [a.shape[0] for a in arrays]

    for i, a in enumerate(arrays):
        if i == 0:
            out = a[0:min(heights), :, :]
        else:
            out = _np.hstack([out, a[0:min(heights), :, :]])
    return out



def check_array(a, b):
    '''(ndarray,ndarray)
    perform array checks raising error if problem
    '''
    if a.shape != b.shape:
        raise ValueError('Array shapes did not match.')

def shape(l):
    '''(list|tuple) -> tuple
    returns shape of a list or a tuple
    by converting it to an np array
    and returning nparray.shape
    '''
    return _np.array(l).shape

def np_permute_2d(a):
    '''(ndarray) -> ndarray
    Takes a numpy array and permutes the values ignoring NaNs
    i.e. the array can contain NaNs but a permuted value
    cannot be permuted into a cell of value NaN
    '''
    assert isinstance(a, _np.ndarray)

    # get a numpy flattened array of all values which are a number (ie exclude
    # NaNs)
    mask = _np.isfinite(a)  # create a boolean mask

    np_val_list = a.copy()
    np_val_list = np_val_list[mask]
    np_val_list = _np.random.permutation(np_val_list)

    # now we need to reassign our original array for the permuted list where there are non NaNs
    # First get indexes of non NaN values in passed array
    # mask is still a array of booleans with True values corresponding to non
    # NaNs and Infs
    np_inds = _np.nonzero(mask)
    np_inds = _np.transpose(np_inds)
    cnt = 0
    npout = a.copy()
    assert isinstance(npout, _np.ndarray)
    for val in np_inds:
        npout[val[0]][val[1]] = np_val_list[cnt]
        cnt += 1

    return npout


def _focal_mean_filter(arg):
    '''(array) -> scalar(float)
    Function used by np_focal_mean by the ndimage.filters.generic_filter
    to calculate per element focal values.
    In particular we want to return NaN when the original element is NaN
    '''
    if _np.isnan(arg[4]):
        return _np.NaN
    return _np.nanmean(arg)


def np_focal_mean(a, pad=True):
    '''(ndarray of 2 dimensions, bool) -> ndarray
    If pad is true, adds a NaN border all around the input array
    Calculates focal mean on elements of numpy array which are not NaN
    Radius is currently all adjacent cells

    May get unexpected results if ndarray is not of type float
    '''
    assert isinstance(a, _np.ndarray)
    x = a.astype(float)

    if pad:
        # surround with nans so we can ignore edge effects
        x = _np.pad(x, pad_width=1, mode='constant', constant_values=_np.NaN)

    kernel = _np.ones((3, 3))

    # create means by kernel
    out = _ndimage.filters.generic_filter(
        x,
        _focal_mean_filter,
        footprint=kernel,
        mode='constant',
        cval=_np.NaN)

    return out


def np_paired_zeros_to_nan(a, b):
    '''(ndarray, ndarray) -> dictionary
    returns  'a':a, 'b':b
    replaces matched zero value pairs with nans, retaining
    the shape of the array
    '''
    assert isinstance(a, _np.ndarray)
    assert isinstance(b, _np.ndarray)
    if a.dtype != 'float':
        raise ValueError('ndarray a is not of dtype float')
    if b.dtype != 'float':
        raise ValueError('ndarray b is not of dtype float')

    if a.shape != b.shape:
        raise ValueError('Arrays must be the same shape')
    nponebool = _np.array(a, dtype=bool)
    nptwobool = _np.array(b, dtype=bool)

    # mask now has False where both 'in' arrays have matching zeros
    # we the invert so matched zero positions are set to True
    # checked and nan is converted to True during above casting
    mask = _np.invert(_np.logical_or(nponebool, nptwobool))

    # npInds now contains indexes of all 'cells' which had zeros
    np_inds = _np.nonzero(mask)
    assert isinstance(np_inds, tuple)

    np_inds = _np.transpose(np_inds)

    for val in np_inds:
        x, y = val
        a[x][y] = _np.NaN
        b[x][y] = _np.NaN

    return {'a': a, 'b': b}


def np_pad_nan(a):
    '''(ndarray) -> ndarray
    pads nd with nans'''
    if a.dtype != 'float':
        raise ValueError('ndarray is not of dtype float')
    return _np.pad(a, pad_width=1, mode='constant', constant_values=_np.NaN)


def np_delete_paired_nans_flattened(a, b):
    '''(ndarray, ndarray) -> ndarray
    Array types are float
    This must first flatten both arrays and both outputs
    are FLATTENED (but retain matches at a given index)
    {'a':a, 'b':b}
    '''
    assert isinstance(a, _np.ndarray)
    assert isinstance(b, _np.ndarray)
    if a.shape != b.shape:
        raise ValueError('arrays are of different shape')

    a = a.flatten()
    b = b.flatten()

    # set mask values to false where there are nans
    # then use mask for both a and b to filter out all matching
    # nans
    a = a.astype(float)
    b = b.astype(float)

    amask = _np.invert(_np.isnan(a))
    bmask = _np.invert(_np.isnan(b))
    mask = _np.logical_or(amask, bmask)
    a = a[mask]
    b = b[mask]

    return {'a': a, 'b': b}


def np_nans_to_zero(a):
    '''(ndarray, ndarray) -> dict
    Where there are unmatched nans by position in ndarrays
    a and b, zero will be substituted.
    a and b will be converted to dtype=float
    returns {'a':a,'b':b}
    '''
    assert isinstance(a, _np.ndarray)

    out = a.copy().astype(float)
    mask = numpy.isnan(out)
    # inds where isnan is true, looks like [(11,1),(5,4) ...]
    inds = _np.nonzero(mask)
    inds = zip(inds[0], inds[1])
    for x, y in inds:
        if _np.isnan(out[x][y]):
            out[x][y] = 0

    return out


def np_round_extreme(a):
    '''(ndarray) -> ndarray
    Rounds negative numbers to be more negative int
    and positve numbers to be more positive int
    '''
    tmp = _np.copy(a)
    tmp[tmp < 0] = _np.floor(tmp[tmp < 0])
    tmp[tmp > 0] = _np.ceil(tmp[tmp > 0])
    return tmp


def np_unmatched_nans_to_zero(a, b):
    '''(ndarray, ndarray) -> dict
    Where there are unmatched nans by position in ndarrays
    a and b, zero will be substituted.
    a and b will be converted to dtype=float
    returns {'a':a,'b':b}
    '''
    assert isinstance(a, _np.ndarray)
    assert isinstance(b, _np.ndarray)
    if a.shape != b.shape:
        raise ValueError('Arrays must be same shape')

    a = a.astype(float)
    b = b.astype(float)
    mask = np_unmatched_nans(a, b)

    # this gets the indexes of cells with unmatched nans
    inds = _np.nonzero(mask)
    # inds looks like [(11,1),(5,4) ...]
    inds = zip(inds[0], inds[1])
    for ind in inds:
        if _np.isnan(a[ind[0]][ind[1]]):
            a[ind[0]][ind[1]] = 0
        else:
            b[ind[0]][ind[1]] = 0

    return {'a': a, 'b': b}


def np_unmatched_nans(a, b):
    '''(ndarray, ndarray) -> ndarray
    Creates a new array where nans do not match position in each array

    nan<->nan = False
    nan<->1.2 = True
    1.2<->1.2 = False

    Arrays must be of the dimensions

    Returned ndarray has True where nans are unmatched
    '''
    assert isinstance(a, _np.ndarray)
    assert isinstance(b, _np.ndarray)
    if a.shape != b.shape:
        raise ValueError('Arrays must be same shape')

    a = a.astype(float)
    b = b.astype(float)

    amask = _np.isnan(a)
    bmask = _np.isnan(b)
    mask = _np.logical_xor(amask, bmask)
    return mask


def np_delete_paired_zeros_flattened(a, b):
    '''(ndarray, ndarray) -> dictionary
    'dic is 'a':aOut, 'b':bOut
    This must first flatten both arrays and both outputs
    are flattened (but retain matches at a given index
    '''
    assert isinstance(a, _np.ndarray)
    assert isinstance(b, _np.ndarray)
    a = a.flatten()
    b = b.flatten()

    # set mask values to false where there are zeros
    # then use mask for both a and b to filter out all matching
    # zeros
    amask = _np.invert(a == 0)
    bmask = _np.invert(b == 0)
    mask = _np.logical_or(amask, bmask)
    return {'a': a[mask], 'b': b[mask]}


def angles_between(vectors1, vectors2):
    '''(ndarray|list|tuple, ndarray|list|tuple) -> ndarray
    Get pairwise angles between an array of vectors.

    This is vectors1 is broadcast across vectors2.

    vectors1, vectors2:
        arrays of vectors (i.e. 1-nested list likes

    Returns:
        Numpy array of angles
    '''
    v1 = makenp(vectors1)
    v2 = makenp(vectors2)
    costheta = 1 - dist.cdist(v1, v2, 'cosine')
    return _np.arccos(costheta)



def makenp(in_):
    '''(ndarray|list|tuple|set) -> ndarray
    convert list type to numpy
    array, or return a copy
    if in_ was already a numpy array
    '''
    if isinstance(in_, _np.ndarray):
        return _np.copy(in_).astype(_np.float)

    if isinstance(in_, (tuple, list, set)):
        return _np.asarray(in_, dtype=float) #forcing to float will handle None values

    raise ValueError('Expected tuple, list or set. Got %s' % type(in_))


def distances(origs, dests):
    '''(ndarray|list|tuple, ndarray|list|tuple) -> ndarray
    Create 2d array of distance between n-dimensional points, i.e.
    creates an n x m matrix of distances between each point rathern
    than a pairwise set of distances.

    origs is broadcasted to dests

    Format for points is:
    [[0,0]] - a single 2D point at the origin
    [[0,0,0]] - a single 3D point at the origin

    Examples:
    >>>a=np.array([[0,0,0]])
    >>>b=np.array([[1,1,1],[2,2,2]])
    >>>arraylib.distances(a,b)
    array([[1.73205081, 3.46410162]])

    >>>a=np.array([[0,0,0],[0,0,0]])
    >>>b=np.array([[1,1,1],[2,2,2]])
    >>>arraylib.distances(a,b)
    array([[1.73205081, 3.46410162], [1.73205081, 3.46410162]])
    '''
    nd_o = makenp(origs)
    nd_d = makenp(dests)
    subts = nd_o[:,None,:] - nd_d
    return _np.sqrt(_np.einsum('ijk,ijk->ij',subts, subts))


def np_delete_zeros(a):
    '''(arraylike) -> ndarray
    delete zeros from an array.
    **Note that this will reshape the array**
    '''
    a = _np.array(a).astype(float)
    _np.place(a, a == 0, _np.nan)
    return np_delete_nans(a)


def np_delete_nans(a):
    '''(arraylike) -> ndarray

    Takes an array like and removes all nans.

    **Note that this will change the location of values in the array**
    '''
    nd = _np.array(a).astype(float)
    return nd[_np.invert(_np.isnan(nd))]


def np_contains_nan(nd):
    '''(ndarray) -> bool
    Global check if array contains np.nan anywhere
    '''
    return _np.isnan(_np.sum(nd))


def np_pickled_in_excel(pickle_name):
    '''(str, bool) -> void
    opens the pickled nd array as a new excel spreadsheet

    If silent_save is true, then the file is saved as an excel file
    to the same directory (and name) as the pickled nd array

    Currently assumes a 1D or 2D array. Unknown behaviour with >2 axis.
    '''
    arr = _np.load(pickle_name)
    try:
        import xlwings
        xlwings.view(arr)
    except Exception as _:
        _warn('np_pickled_in_excel not supported because of xlwings dependency')


def max_indices(arr, k):
    '''(ndarray|list|tuple, int) -> list

    Returns the indices of the k first largest elements of arr
    (in descending order in values)

    Example:
    >>>max_indices([1,4,100,10], 2)
    [3, 4]
    '''
    arr_ = makenp(arr)
    assert k <= arr_.size, 'k should be smaller or equal to the array size'
    arr_ = arr_.astype(float)  # make a copy of arr
    max_idxs = []
    for _ in range(k):
        max_element = _np.nanmax(arr_)
        if _np.isinf(max_element):
            break
        else:
            idx = _np.where(arr_ == max_element)
        max_idxs.append(idx[0].tolist())
        arr_[idx] = -_np.inf
    out = _list_flatten(max_idxs)
    return out


def min_indices(arr, k):
    '''(ndarray|list|tuple, int) -> list

    Returns the indices of the k first largest elements of arr
    (in descending order in values)

    Example:
    >>>max_indices([1,4,100,10], 2)
    [3, 4]
    '''
    arr_ = makenp(arr)

    assert k <= arr_.size, 'k should be smaller or equal to the array size'
    arr_ = arr_.astype(float)  # make a copy of arr
    arr_[arr_ == 0.] = _np.inf
    min_idxs = []
    for _ in range(k):
        min_element = _np.nanmin(arr_)
        if _np.isinf(min_element):
            break
        else:
            idx = _np.where(arr_ == min_element)
        min_idxs.append(idx[0].tolist())
        arr_[idx] = _np.inf
    out = _list_flatten(min_idxs)
    return out


def np_frequencies(a):
    '''(ndarray)->ndarray
    return array with frequency values of items in array
    a = [1,2,3,3,4,4,4]
    np_frequencies(a) returns
    [[1 1],
    [2,1],
    [3,2],
    [4,3]]
    '''
    unq, cnt = _np.unique(a, return_counts=True)
    return _np.asarray((unq, cnt)).T


def np_difference(a, b):
    '''(ndarray, ndarray) -> ndarray
    get absolute difference between two matrices.
    Effectively one from the other then abs it.
    '''
    x = _np.copy(a)
    y = _np.copy(b)
    return _np.abs(x - y)


def np_conditional_array_split(a, has_by_column, has_by_row):
    '''(ndarray, bool, bool)->ndarray, ndarray, ndarray
    Given an array of conditional probabilities returns
    marginals and the conditionals as seperate matrices
    [body, col_marginals, row_marginals]
    '''

    rows = int(a.shape[0])
    cols = int(a.shape[1])

    if has_by_column and has_by_row:
        body = a[0:cols - 1, 0:rows - 1]
        row_marginals = a[0:rows - 1, cols - 1:cols]
        col_marginals = a[rows - 1:rows, 0:cols - 1]
    elif has_by_row:
        body = a[0:rows, 0:cols - 1]
        row_marginals = a[0:rows, cols - 1:cols]
        col_marginals = []
    elif has_by_column:
        body = a[0:rows - 1, 0:cols]
        col_marginals = a[rows - 1:rows, :]
        row_marginals = []
    return [body, col_marginals, row_marginals]
# endregion


# region Pandas
def pd_df_to_ndarray(df):
    '''(dataframe)->ndarray
    Return a dataframe as a numpy array
    '''
    return df.as_matrix([x for x in df.columns])
# endregion



def np_split_by_value(a, thresh):
    '''(ndarray, float|bool) -> ndarray, ndarray
    split an array into two arrays at thresh

    Example:
    >>>np_split_by_value(np.array([1,2,3]), 2)
    ([1,2]), ([3])
    '''
    if isinstance(thresh, bool):
        Z = a == True
    else:
        Z = a <= thresh
    return a[Z == 0], a[Z == 1]
