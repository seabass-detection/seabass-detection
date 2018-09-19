# pylint: disable=C0103, too-few-public-methods, locally-disabled,
# no-self-use, unused-argument
'''Distance measures'''
import numpy as _np

__all__ = ['L1dist', 'L2dist']


def L2dist(p1, p2):
    '''(ndarray,ndarray)->float
    Euclidean L2 distance between two points.
    eg.
    A=np.array([[0,0]])
    B=np.array([[1,1]])
    L2dist(A, B) = 1.4142...
    '''
    if isinstance(p1, (list, tuple)):
        return _np.sqrt(_np.sum((_np.array(p1) - _np.array(p2))**2))

    return _np.sqrt(_np.sum((p1 - p2)**2))


def L1dist(v1, v2):
    '''(ndarray,ndarray)->float
    L1dist (minkowski/taxi cab) distance between 2 points
    A=np.array([[0,0]])
    B=np.array([[1,1]])
    L2dist(A, B) = 2
    '''
    return _np.sum(abs(v1 - v2))


def nearN_euclidean(point, points, nr=1):
    '''(ndarray|list|tuple|ndarray|list|tuple,int)->list [[ind1,dist1]int (index of tuple), point of same type as points
    Given a point and an array of points
    return a list of lists containing the index and distances of the nr closest points to point.
    eg:
    pt = [0,0]: pts=[[0,0], [1,1], [3,3]]
    nearN_euclidean(point, points, nr=2)
    # ([0,0], [1,1.414

    0 and 1 are the indices and 1, 1.414 are the distances
    '''
    ndpt = _np.array(point)
    ndpts = _np.array(points)
    diff = ndpts - ndpt
    dist = diff*diff
    dist = _np.sqrt(dist[0:, 0] + dist[0:, 1])

    nr = len(dist) if nr > len(dist) else nr #argpartition generates an error if nr greater than nr of elements in dist
    ret = _np.argpartition(dist, -nr)[:nr]
    l_ind = ret.tolist()
    l_dist = dist[l_ind].tolist()
    out = [x for x in zip(l_ind, l_dist)]
    return out


def furthestN_euclidean(point, points, nr=1):
    '''(ndarray|list|tuple|ndarray|list|tuple,int)->list [[int,float], ... ]
    Given a point and an array of points
    return a list of lists of the index and distance of the nr most distant points from point.
    eg:
    >>> pt = [0,0]; pts=[[0,0], [1,1], [3,3]]
    >>> furthestN_euclidean(pt, pts, nr=2)
    [[2,9], [1,1.414]]

    2 and 1 are the indices and 9 and 1.414 are the distances

    This list is NOT ordered, i.e. the first in the list may not be the furthest,
    unless ofcouse nr=1
    '''
    assert len(points) >= nr, 'Points should be greater than or equal to nr'
    ndpt = _np.array(point).astype('float')
    ndpts = _np.array(points).astype('float')

    diff = ndpt - ndpts
    dist = diff*diff
    dist = _np.sqrt(dist[0:, 0] + dist[0:, 1])
    if len(points) == 1:
        out = [[0, dist[0]]]
    else:
        ret = _np.argpartition(dist, -nr)[-nr:]
        l_ind = ret.tolist()
        l_dist = dist[l_ind].tolist()
        out = [x for x in zip(l_ind, l_dist)]
    return out


def nearestN_euclidean(point, points, nr=1):
    '''(ndarray|list|tuple|ndarray|list|tuple,int)->list [[int,float], ... ]
    Given a point and an array of points
    return a list of lists of the index and distance of the nr most distant points from point.
    eg:
    >>> pt = [0,0]; pts=[[0,0], [1,1], [3,3]]
    >>> furthestN_euclidean(pt, pts, nr=2)
    [[2,9], [1,1.414]]

    2 and 1 are the indices and 9 and 1.414 are the distances

    This list is NOT ordered, i.e. the first in the list may not be the nearest,
    unless ofcouse nr=1
    '''
    assert len(points) >= nr, 'Points should be greater than or equal to nr'
    ndpt = _np.array(point).astype('float')
    ndpts = _np.array(points).astype('float')
    diff = ndpts - ndpt
    dist = diff*diff
    dist = _np.sqrt(dist[0:, 0] + dist[0:, 1])
    if len(points) == 1:
        out = [[0, dist[0]]]
    else:
        ret = _np.argpartition(dist, nr)[:nr]
        l_ind = ret.tolist()
        l_dist = dist[l_ind].tolist()
        out = [x for x in zip(l_ind, l_dist)]
    return out


def linear_distance_matrix(r, c):
    '''(int, int) -> ndarray
    Returns a distance matrix based on linear distance.

    args
        r: row count
        c: column count

    Example:
    linear_distance_matrix(3,3)

      0 1 2
    -------
    0|0 1 2
    1|1 0 1
    2|2 1 0
    '''
    return _np.fromfunction(lambda i, j: abs(i - j), (r, c), dtype=int)
