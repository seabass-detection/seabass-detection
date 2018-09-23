# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
'''working with opencv shapes'''
import math as _math

from sympy import Polygon, Point2D

import numpy as _np
from numpy.random import uniform
from scipy.optimize import fsolve as _fsolve

import funclib.baselib as _baselib
import opencvlib.distance as _dist

RADIANS45 = 0.7853981633974483
RADIANS90 = 1.5707963267948966
RADIANS60 = 1.0471975511965976

def get_rnd_pts(range_=(-50, 50), n=10, dtype=_np.int):
    '''(2-tuple, 2-tuple, int, class:numpy.dtype) -> ndarray
    Returns an n x 2 ndarray of unique random points
    '''

    if isinstance(dtype, _np.int):
        range_ = (range_[0], range_[1] + 1)

    return _np.reshape(uniform(range_[0], range_[1], n*2), (n, 2)).astype(dtype)


def triangle_pt(pt1, pt2, ret_max_y_pt=True):
    '''(2-array, 2-array, bool)-> 2-list
    Given a line defined by pt1 and pt2, get the third point
    to make a right angled triangle.

    Coordinates are CVXY

    pt1:
        first line point
    pt2:
        second point
    ret_max_y_pt:
        return the point with the maximum y coordinate,
        otherwise returns the point with the minimum y coordinate

    returns:
        2-list, representing the third point
    '''
    assert len(pt1) == len(pt2) == 2, 'pt1 and pt2 should be 2 elements array likes.'
    y1 = pt1[1]
    y2 = pt2[1]
    x1 = pt1[0]
    x2 = pt2[0]

    pt3 = [x2, y1]
    pt4 = [x1, y2]

    if pt3[1] > pt4[1]:
        return pt3 if ret_max_y_pt else pt4

    return pt3 if not ret_max_y_pt else pt4


def angle_between_pts(pt1, pt2, as_degrees=True):
    '''(array, array, bool, bool) -> float
    Calculate the angle between two points
    '''
    ang1 = _np.arctan2(*pt1[::-1])
    ang2 = _np.arctan2(*pt2[::-1])
    if as_degrees:
        return _np.rad2deg((ang1 - ang2) % (2 * _np.pi))

    return (ang1 - ang2) % (2 * _np.pi)


def angle_min_rotation_to_x(angle, as_degrees=True):
    '''(float) -> float
    Get the minimum rotation to the x-axis from angle,
    where angle is the rotation to x axis, with negative
    being clockwise.

    angle:
        the angle, e.g -180 would be a line with coords ((0,0), (-1,0))
        this line would be parralel to the x-axis, rotating -180 would
        be incorrect in most circumstances, it should be rotated 0 degrees.

    Example:
    >>> angle_min_rotation_to_x(-45)
    -45
    >>> angle_min_rotation_to_x(-300)
    60 #anticlockwise
    >>> angle_min_rotation_to_x(-190)
    -10 #clockwise
    '''
    if -270 <= angle <= -90:
        a = 180 - abs(angle)
    elif angle == -360:
        a = 0
    elif -360 < angle < -270:
        a = 360 - abs(angle)
    else:
        a = angle

    if not as_degrees:
        a = _math.radians(a)

    return a




def length_between_pts(pts, closed=False):
    '''(array, bool) -> float
    Length of lines defined by points

    pts:
        iterable, in format [[1,2], [3,4], ...]
    closed:
        include length between first and last point

    Example:
    >>> length_between_pts([[0,0],[1,1]])
    1.4142135623730951
    '''
    l = 0

    if len(pts) <= 1:
        return 0

    for i in range(0, len(pts) - 1):
        l += _dist.L2dist(pts[i], pts[i+1])

    if closed:
        l += _dist.L2dist(pts[0], pts[-1])
    return l


def length_between_all_pts(pts):
    '''(array, bool) -> list
    Return all lengths between points in
    the order they appear

    pts:
        iterable, in format [[1,2], [3,4], ...]

    Example:
    >>> length_between_pts([[0,0],[1,1]])
    1.4142135623730951, 1.4142135623730951
    '''
    l = []

    if len(pts) <= 1:
        return 0

    for i in range(0, len(pts) - 1):
        l.append(_dist.L2dist(pts[i], pts[i+1]))

    l.append(_dist.L2dist(pts[0], pts[-1]))

    return l


def rotation_angle(pt1, pt2, as_radians=False):
    '''(2-array, 2-array, bool) -> float
    Get the angle a line (defined by two points) needs
    to be rotated through to be parallel with the x-axis

    Intended to find the rotation required for a
    'standard' line has been found.

    Assumes CVXY frame

    pt1:
        first point
    pt2:
        second point
    as_radians:
        return as radians, not angle

    returns:
        the angle

    Notes:
    Transforms.rotate takes an angle, not radians. Although the
    frame is interpreted as standard xy, this works out as a
    negative anticlockwise rotation.
    '''
    max_y = triangle_pt(pt1, pt2)[1]
    pt = _baselib.list_flatten([x for x in [pt1, pt2] if x[1] == max_y])
    T = [x*-1 for x in pt] #get the translation needed to move pt to the origin
    pt_to_transform = pt1 if pt2 == pt else pt2 #get the other point of the line
    rad = _math.atan2(pt_to_transform[1] + T[1], pt_to_transform[0] + T[0]) #calculate the angle after translation to the origin
    return rad if as_radians else _math.degrees(rad)


def rotate_points(pts, angle, center=(0, 0), translate=(0, 0)):
    '''(ndarray|list, float, 2-tuple|None, 2-tuple) -> nx2-tuple
    Rotates a point "angle" degree around center.
    Negative angle is clockwise.

    pts: array like of len 2, eg [[1,2],[2,3], ...]
    angle: angle to rotate in degrees (e.g. -90)
    center:
        point around which to rotate, (x, y),
        if center is None, rotate about the
        center of the points
    translate:
        Add a translation, this is useful if an image
        has been rotated without croping. In this case
        we add a positive y and x translation of
        translate = (
                    (img_rotated.shape[0] - img_orig.shape[0])/2,
                    (img_rotated.shape[1] - img_orig.shape[1])/2
                    )
    Example:
    >>>rotate_points([[10,10],[20,20]], -45, center=(15,15))
    [(15.0, 7.9289321881345245), (15.0, 22.071067811865476)]
    '''
    out = []
    if not center:
        center = _np.mean(pts, axis=0)
    out = [rotate_point(pt, angle, center, translate) for pt in pts]
    return out


def rotate_point(pt, angle, center=(0, 0), translate=(0, 0)):
    '''(2-array, float, 2-tuple) -> 2-tuple
    Rotates a point "angle" degree around center.
    Negative angle is clockwise.

    pt:
        array like of len 2, eg [1,2]
    angle:
        angle to rotate in degrees (e.g. -90)
    center:
        point around which to rotate, (x, y)
    translate:
        Add a translation, this is useful if an image
        has been rotated without croping. In this case
        we add a positive y and x translation of
        translate = (
                    (img_rotated.shape[0] - img_orig.shape[0])/2,
                    (img_rotated.shape[1] - img_orig.shape[1])/2
                    )
    '''
    angle = -1*angle #the angle as passed will be negative for clockwise, but this routine uses positive for clockwise - make it behave the same
    angle_rad = _math.radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_pt = (pt[0] - center[0], pt[1] - center[1])
    new_pt = (new_pt[0] * _math.cos(angle_rad) - new_pt[1] * _math.sin(angle_rad),
                 new_pt[0] * _math.sin(angle_rad) + new_pt[1] * _math.cos(angle_rad))
    # Reverse the shifting we have done
    return (new_pt[0] + center[0] + translate[0], new_pt[1] + center[1] + translate[1])


def rescale_points(pts, downsample, as_int=False):
    '''(n,2-list, float) -> n,2-list
    Calulates coordinates of points in a notional image
    where that image has been downsampled by factor downsample

    pt: array like of len 2, eg [1,2]
    downsample: downsample factor, new image size will be imagesize/downsample
                hence downsample=2 will 1/2 the length and width

    Example:
    >>> rescale_points([[10,10],[50,50],[100,100]],2)
    [(5.0, 5.0), (25.0, 25.0), (50.0, 50.0)]
    '''
    out = []
    out = [rescale_point(pt, downsample, as_int) for pt in pts]
    return out


def rescale_point(pt, downsample, as_int=False):
    '''(2-array, float) -> 2-tuple
    Calulates coordinate of a point in a notional image
    where that image has been downsampled by factor downsample.

    pt: array like of len 2, eg [1,2]
    downsample: downsample factor, new image size will be imagesize/downsample
                hence downsample=2 will 1/2 the length and width
    as_int: return as integer, else float

    Example:
    >>>rescale_point((100,100),2.222,True)
    (45, 45)
    '''
    f = 1 / downsample
    pt_ = (f*pt[0], f*pt[1])
    if as_int:
        pt_ = (int(f*pt[0]), int(f*pt[1]))
    return pt_


def flip_points(pts, h, w, hflip=True):
    '''(n,2-ndarray, float, float, bool) -> 2-tuple
    Flip a point horizontally or vertcally

    pts:
        array like of len 2, eg [[1,2],[2,3], ...]
    h,w:
        Image height and width in pixels
    hflip:
        Flip horizontally, otherwise vertically
    '''
    out = []
    assert isinstance(h, (float, int))
    assert isinstance(w, (float, int))
    assert h > 0, 'Image height must be > 0'
    assert w > 0, 'Image width must be > 0'

    out = [flip_point(pt, h, w, hflip) for pt in pts]
    return out


def flip_point(pt, h, w, hflip=True):
    '''(2-array, float, float, bool) -> 2-tuple
    Flip a point horizontally or vertcally

    pt:
        array like of len 2, eg [1,2]
    h,w:
        Image height and width in pixels
    hflip:
        Flip horizontally, otherwise vertically
    '''
    if hflip:
        return (w - pt[0], pt[1])

    return (pt[0], h - pt[1])


def centroid(pts, dtype=_np.float):
    '''(ndarray|list|tuple) -> 2-list
    Get centroid of pts as 2-list

    pts:
        n x 2 list like

    Example:
    >>> centroid([[0,0],[10,10]])
    [5.0, 5.0]
    '''
    ndpts = _np.asarray(pts)
    mn = _np.mean(ndpts, axis=0)
    return mn.astype(dtype).tolist()


def valid_point_pairs(pts1, pts2):
    '''(n,2-list, n,2-list) -> n,2-list, n,2-list
    Build matched array of points

    Returns 2 arrays of points from pts1, pts2
    where neither point contains None.

    i.e. it filters out points which are invalid
    returning only those points which are pairwise
    valid.

    If no points found, returns [],[]

    pts1, pts2: n,2-list of points

    e.g.
    >>>pts1, pts2 = build_matched([[None, 1], [10,20]], [[1, 1], [1,2]])
    >>>print(pts1, pts2)
    [[10,20]], [[1,2]]
    '''
    pt1_out = []; pt2_out = []
    for i in range(min([len(pts1), len(pts2)])):
        if not None in pts1[i] and not None in pts2[i]:
            pt1_out.append(pts1[i])
            pt2_out.append(pts2[i])
    return pt1_out, pt2_out


def points_rmsd(V, W):
    '''(ndarray|list|tuple, ndarray|list|tuple) -> float
    Calculates the Root mean square dev between
    two sets of n-dimensional points.

    V, M:
        array like representations of n-D points.
        Should ignore pairwise points where values are None
        or np.nan
    '''
    D = len(V[0])
    N = len(V)
    rmsd = 0.0
    for v, w in zip(V, W):
        v = _np.array(v).astype(_np.float)
        w = _np.array(w).astype(_np.float)
        if True in _np.isnan(v) or True in _np.isnan(w):
            continue
        rmsd += sum([(v[i] - w[i]) ** 2.0 for i in range(D)])

    return _np.sqrt(rmsd/N)



def order_points(p):
    '''(list|tuple|ndarray) -> list
    Sorts a list of points by their position.
    '''
    if isinstance(p, (list, tuple, set)):
        pts = _np.array(p)
    pts = pts.tolist()
    cent = (sum([p[0] for p in pts])/len(pts), sum([p[1] for p in pts])/len(pts))
    pts.sort(key=lambda p: _math.atan2(p[1] - cent[1], p[0] - cent[0]))
    return pts


def rect_side_lengths(pts):
    '''(n,2-list) -> float, float
    Get side lengths of rectangle

    pts: list of points [[0,0],[10,10],[10,0],[0,10]]

    Returns: short side, long side

    Example:
    >>> rect_side_lengths([[0,0],[10,10],[10,0],[0,10]])
    10, 10
    '''
    assert len(pts) == 4, 'Rectangle must have 4 points, got %s' % len(pts)
    ls = length_between_all_pts(pts)
    #this is necessary because if points are in wrong order, can get diagonal
    ls = list(set(ls))
    ls.sort()
    return ls[0], ls[1]


def rect_side_lengths2(pts):
    '''(n,2-list) -> float, float, float
    Get side lengths and diagonal of rectangle

    pts: list of points [[0,0],[10,10],[10,0],[0,10]]

    Returns: short side, long side, diagonal

    Example:
    >>> rect_side_lengths([[0,0],[10,10],[10,0],[0,10]])
    10, 10, 14.142135623730951
    '''
    short, long = rect_side_lengths(pts)
    return _math.sqrt(short**2 + long**2)


def bound_poly_rect_side_length(pts, angle, radians=False, centre=None):
    '''(n,2-list, float, 2-tuple|None, bool) -> 2n-list, float, float

    Rotate a polygon, then get the points and
    side lengths of the bounding rectangle.
    Also orders the returned points.

    pts: n2-list, [[0,0], [10,10], ...]
    angle: angle to rotate
    radians: if true, angle is assumed to be radians, else degrees
    centre: rotate around this centre, if none, rotation is around
            the polygon centre

    Returns: points, short length, long length
    '''
    if radians:
        angle = _math.degrees(angle)
    sqrot = rotate_points(pts, angle, None)
    sqbnd = order_points(bounding_rect_of_poly2(sqrot))
    sq_b, sq_a = rect_side_lengths(sqbnd)
    return sqbnd, sq_b, sq_a


def bound_poly_rect_side_length2(pts, angle, radians=False, centre=None):
    '''(n,2-list, float, 2-tuple|None, bool) -> 2n-list, float, float

    Rotate a polygon, then get the points and
    side lengths of the bounding rectangle for the
    second model of rotation of boundng box detection.

    Also orders the returned points.

    pts: n2-list, [[0,0], [10,10], ...]
    angle: angle to rotate
    radians: if true, angle is assumed to be radians, else degrees
    centre: rotate around this centre, if none, rotation is around
            the polygon centre

    Returns: points, short length, long length

    Example:
    >>>print(bound_poly_rect_side_length2([[0, 0], [0, 5], [20, 5], [20, 0]], 10))
    [[0.4341204441673252, -1.6985011591998234], [20.130275504411486, -1.6985011591998234], [20.130275504411486, 6.698501159199823], [0.4341204441673252, 6.698501159199823]], 8.397002318399647, 19.69615506024416
    '''
    if radians:
        angle = _math.degrees(angle)
    sqrot = rotate_points(pts, angle, None)
    xs, ys = list(zip(*sqrot))
    xs = list(xs); ys = list(ys)

    xs.sort(); ys.sort()

    h = max(ys) - min(ys)
    x_gap = (xs[-1] - xs[-2]) / 2
    w = xs[-1] - xs[0] - (2 * x_gap)

    pts_out = rect_as_points(ys[0], xs[0] + x_gap, w, h)
    sqbnd = order_points(pts_out)
    return sqbnd, min([h, w]), max([h, w])


def bounding_rect_of_poly2(points, as_points=True, round_=False):
    '''(list|ndarray, bool, bool)->list
    Return points of a bounding rectangle in opencv point format if
    as_points=True.

    Note opencv points have origin in top left

    as_points: if false, returns as a tuple (x,y,w,h), else [[0,0], ...]
    round_: rounds points, else returns as float
    '''
    pts_x, pts_y = list(zip(*points))

    if round_:
        pts_x = [int(x) for x in pts_x]
        pts_y = [int(y) for y in pts_y]

    y = min(pts_y)
    x = min(pts_x)
    h = max(pts_y) - min(pts_y)
    w = max(pts_x) - min(pts_x)

    if as_points:
        return rect_as_points(y, x, w, h)

    return (x, y, w, h)


def rect_as_points(rw, col, w, h):
    '''(int,int,int,int)->list
    Given a rectangle specified by the top left point
    and width and height, convert to a list of points

    rw:
        the y coordinate, origin at the top of the image
    col:
        the x coordinate
    w:
        width of rectangle in pixels
    h:
        height of rectangle in pixels

    returns:
        Points in CVXY format [[x,y], [x+w, y], [x+w, y+h]

    Note:
        The order is top left, top right, bottom right, bottom left.
        This order allows lines to be drawn to connect the points
        to draw as a rectangle.
    '''
    return [(col, rw), (col + w, rw), (col + w, rw + h), (col, rw + h)]


def rect_inner_side_length(pts_outer, ratio, as_radians=True):
    '''(n,2-list, float, bool) -> float, float, float

    Given an outer bounding rectangle, estimate the
    sides and rotation angle of the inner bounding rectangle
    given the predicted ratio of the inner bounding sides.

    pts: n,2-list of points
    ratio: the ratio of the long side/short side (i.e. ratio > 1)
    as_radians: return predicted rotation as radians, else degrees

    Returns: short length, long length, rotation angle in radians

    Notes:
        the rotaation angle may be 90 - angle

    Example:
    >>>rect_inner_side_length([[0,12.5],[5,12.5],[5,0],[0,0]], 2.5)
    4.999999999919968, 1.5184364491905264
    '''
    assert len(pts_outer) == 4, 'pts_outer should have 4 points, found %s' % len(pts_outer)
    A, B = rect_side_lengths(pts_outer)
    short_side, theta = _fsolve(_inner_box_b, (float(min(A, B)), 0.), args=(A, B, ratio)) #, diag=(1, 0.1), maxfev=100000
    ab_ratio = ratio if ratio > 1 else 1 / ratio
    long_side = float(short_side)*ab_ratio
    theta = theta if as_radians else _math.degrees(theta)
    return short_side, long_side, theta


def rect_inner_side_length2(pts_outer, ratio, as_radians=True):
    '''(n,2-list, float, bool) -> float, float, float
    Second rotation model to estimate rectangle side lengths
    from outer bounding  rectangle.

    Given an outer bounding rectangle, estimate the
    sides and rotation angle of the inner bounding rectangle
    given the predicted ratio of the inner bounding sides.

    pts: n,2-list of points
    ratio: the ratio of the long side/short side (i.e. ratio > 1)
    as_radians: return predicted rotation as radians, else degrees

    Returns: short length, long length, rotation angle in radians

    Notes:
        the rotaation angle may be 90 - angle

    Example:
    >>>rect_inner_side_length2([[0,12.5],[5,12.5],[5,0],[0,0]], 2.5)
    '''
    assert len(pts_outer) == 4, 'pts_outer should have 4 points, found %s' % len(pts_outer)
    A, B = rect_side_lengths(pts_outer)
    short_side, theta = _fsolve(_inner_box_b2, (float(min(A, B)), 0.0001), args=(B, A, ratio)) #, diag=(1, 0.1), maxfev=100000
    ab_ratio = ratio if ratio > 1 else 1 / ratio
    long_side = float(short_side)*ab_ratio
    theta = theta if as_radians else _math.degrees(theta)
    return short_side, long_side, theta


def pt_in_poly(pt, poly, order=True):
    '''(2-list,  n,2-list, bool)-> bool
    Does pt lie within poly
    '''
    if order:
        poly_ = order_points(poly)
    else:
        poly_ = list(poly)
    pt = Point2D(pt)
    polygon = Polygon(*poly_)
    return polygon.encloses_point(pt)


def pts_in_poly(pts, poly):
    '''(n,2-list, n,2-list) -> bool
    Are all points in poly.

    >>>pts_in_poly([[1,1],[2,2],[1,2],[2,1]],[[0,0],[0,10],[10,10],[10,0]]
    True
    '''
    poly_ = order_points(poly)
    return all(list(pt_in_poly(pt, poly_, order=False) for pt in poly_))


def _inner_box_b(b_theta, *args):
    '''(2-tuple, 3-tuple) -> float, float
    Get width and angle of rotation
    of inner rect from known outer
    rect width and height

    b_theta: 2-tuple, (starting values we are solving for, i.e. height and theta)
    args: 3-tuple, (W,H, wh_ratio): Width, height of detection box and the ideal ratio of a detection)

    Returns:
        predicted width and the rotation
    '''
    A, B, ab_ratio = args
    b, theta = b_theta
    xx = b * _math.sin(theta) + ab_ratio * b * _math.cos(theta) - A
    yy = ab_ratio * b * _math.sin(theta) + b * _math.cos(theta) - B
    return (xx, yy)


def _inner_box_b2(b_theta, *args):
    '''(2-tuple, 3-tuple) -> float, float
    Get width and angle of rotation
     of inner rect from known outer
   rect width and height

    b_theta: 2-tuple, (starting values we are solving for, i.e. shortest side and theta)
    args: 3-tuple, (W, H, ab_ratio):
        Width, height of detection box and the
        ratio of the long side to the short side
        of an ideal detection.

    Returns:
        predicted width and the rotation
    '''
    A, B, ab_ratio = args
    b, theta = b_theta


    xx = (b * ab_ratio * _math.cos(theta)) - A
    yy = (b * _math.cos(theta)) + (b * ab_ratio * _math.sin(theta))  - B
    return (xx, yy)
