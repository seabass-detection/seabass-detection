# pylint: disable=C0103, too-few-public-methods, locally-disabled, unused-import
# no-self-use, unused-argument
'''related to getting regions of interest
from an image.

Includes some geometry related functions for shapes
'''
from random import randint as _randint
from enum import Enum as _enum

import cv2 as _cv2
import numpy as _np
from numpy import ma as _ma

import opencvlib as _opencvlib
import opencvlib.info as _info
import opencvlib.distance as _dist
import opencvlib.geom as _geom

import funclib.baselib as _baselib
from funclib.arraylib import np_round_extreme as _rnd
from opencvlib import getimg as _getimg

#as we may expect to find these here as well
from opencvlib.geom import bounding_rect_of_poly2, rect_as_points, flip_points





__all__ = ['bounding_rect_of_ellipse', 'bounding_rect_of_poly', 'poly_area',
           'rect2rect_mtx', 'rect_as_points', 'rects_intersect', 'roi_polygons_get',
           'sample_rect', 'to_rect']


class ePointConversion(_enum):
    '''Enumeration for point coversion between frames
    XYMinMaxtoCVXY is [xmin,xmax,ymin,ymax] to [[x,y], ...]
    '''
    XYtoRC = 0
    XYtoCVXY = 1
    RCtoXY = 2
    RCtoCVXY = 3
    CVXYtoXY = 4
    CVXYtoRC = 5
    XYMinMaxtoCVXY = 6 #[xmin,xmax,ymin,ymax]
    Unchanged = 99


class ePointsFormat(_enum):
    '''
    Output formats of points in an array

    XY:
        [[x1, y1], [x2, y2]]
    eForPolyLine:
        numpy array of shape (pts nr, 1, 2), used for plotting polylines
    XXXX_YYYY:
        [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    minMax:
        [xmin, xmax, ymin, ymax]
    xywh:
        [xmin, ymin, w, h]
    rchw:
        [ymin, xmin, h, w]
    '''
    XY = 0
    ForPolyLine = 1
    XXXX_YYYY = 2 #[[x1, x2, x3, x4], [y1, y2, y3, y4]]
    XYWH = 3
    RCHW = 4



class ePointFormat(_enum):
    '''Format of an individual point'''
    XY = 0
    CVXY = 1
    RC = 2


class Line():
    '''Holds a line
    Angles are from the y axis, which is 0 degrees

    Point formats are CVXY
    '''

    def __init__(self, pt1, pt2):
        '''(array, array)
        pt1 and pt2 are both 2-arrays, they should be
        in CVXY format (i.e. origin at top left, x-coord first)
        '''
        assert len(pt1) == len(pt2) == 2, 'pt1 and pt2 should be 2 elements array likes.'
        self.pt1 = pt1
        self.pt2 = pt2
        self.length = None
        self.angle_to_x = None
        self.angle_min_rotation_to_x = None #smallest rotation to make parallel to x axis
        self.midpoint = None
        self._refresh()


    def __call__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2
        self._refresh()


    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)


    def _refresh(self):
        '''internal function to refresh
        length and angles
        '''
        self.length = _dist.L2dist(self.pt1, self.pt2)
        self.angle_to_x = _geom.rotation_angle(self.pt1, self.pt2)
        self.angle_min_rotation_to_x = _geom.angle_min_rotation_to_x(self.angle_to_x)
        self.midpoint = [(self.pt1[0] + self.pt2[0]) / 2, (self.pt1[1] + self.pt2[1]) / 2]


class Quadrilateral():
    '''represents a quadrilateral shape in opencv
    and provides support for rotations.

    The size of the image must be provided for
    proper rotation.

    Properties:
        lines: list of Line instances
        angle_to_origin:angle required to rotate shape to be parallel with origin
    '''

    def __init__(self, pts, frame_x, frame_y):
        '''(array, int, int)

        pts:
            array like of 4 points e.g. [[0,0],[10,0],[0,10],[10,10]]
        frame_x:
            image width
        frame_y:
            image height
        '''
        self._pts = pts
        self.lines = []
        self.angle_to_origin = None
        self._frame_x = frame_x
        self._frame_y = frame_y
        if pts:
            self._refresh()


    def __call__(self, pts, frame_x=None, frame_y=None):
        self._pts = pts
        if frame_x:
            self._frame_x = frame_x
        if frame_y:
            self._frame_y = frame_y
        self._refresh()


    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return ', '.join(sb)


    def _refresh(self):
        assert len(self._pts) == 4, 'Expected 4 points, got %s.' % len(self._pts)
        for l in range(0, len(self._pts) - 1):
            self.lines.append(Line(self._pts[l], self._pts[l + 1]))
        self.lines.append(Line(self._pts[-1], self._pts[0])) #line joining first and last points
        self._angle()


    def _angle(self):
        '''() -> void
        calculates the angle to rotate the quadrilateral to
        make it parralel with the x axis, setting the class property
        angle_to_origin

        It doe this by finding the line joining the midpoint of the two shortest sides and
        and calculating the angle of this line to the x-axis using arctan.
        '''
        #get two shortest sides
        assert len(self.lines) == 4, 'Expected 4 line objects in quadrilateral object, have you initialised it properly?'
        dic = {i:ln.length for i, ln in enumerate(self.lines)}
        s = _baselib.dic_sort_by_val(dic)

        line1 = self.lines[s[0][0]]
        line2 = self.lines[s[1][0]]
        assert isinstance(line1, Line)
        assert isinstance(line2, Line)

        #get the angle from the two shortest sides as defined by their points
        self.angle_to_origin = _geom.rotation_angle(line1.midpoint, line2.midpoint)


    @property
    def rotated_to_x(self, as_int=True):
        '''(bool) -> array
        Returns the quadrilateral points, rotated so the
        shape is parralel with the origin along its long
        axis

        as_int:
            return points as integers,

        Returns:
            array of points in CVXY format
        '''
        ret = roi_rotate(self._pts, self.angle_to_origin, self._frame_x/2, self._frame_y/2)
        if as_int:
            ret = _rnd(ret)

        return ret


    @property
    def bounding_rectangle(self):
        '''() -> array

        return the bounding rectangle of the
        rotated_to_x quadrilateral

        Returns:
            array of points in CVXY format
        '''
        pts = self.rotated_to_x
        return bounding_rect_of_poly(pts)



def points_convert(pts, img_x, img_y, e_pt_cvt, e_out_format=ePointsFormat.XY):
    '''(array, 2:tuple, Enum:ePointConversion, Enum:ePointsFormat) -> list
    Converts points in one frame to another.
    XY:Standard cartesian coordinates, RC:Matrix coordinates,
    CVXY:OpenCV XY format which has the Y origin at the top of the image.

    Note that all points are assumed to be in a 0 index.

    pts:
       An array like of points, e.g. [[1,2], [2,3]],
       or [xmin,xmax,ymin,ymax] for XYMinMaxtoCVXY
    img_x:
        image width, e.g. 1024 in a 1024x768 image
    img_y:
        image height, e.g. 768 in a 1024x768 image
    e_pt_cvt:
        the enumeration ePointConversion defining the required conversion
    e_out_format:
        the format of the output points, see ePointsFormat, Note that this just
        the output form - the XY does not imply the order of the axis etc which
        is defined by Enum:ePointConversion.

    returns:
        list of points, [[1,2],[2,3]]
    '''
    #we are using 0 indicies
    img_x -= 1
    img_y -= 1

    out = []
    if e_pt_cvt == ePointConversion.XYMinMaxtoCVXY: #e.g. of points in for this format (10, 50, 20, 30),  i.e. xmin,xmax,ymin,ymax
        out = [[pts[0], pts[2]], [pts[1], pts[2]], [pts[1], pts[3]], [pts[0], pts[3]]]
    else:
        for pt in pts:
            if e_pt_cvt == ePointConversion.XYtoRC:
                out.append([abs(pt[1] - img_y), pt[0]])
            elif e_pt_cvt == ePointConversion.XYtoCVXY:
                out.append([pt[0], abs(pt[1] - img_y)])
            elif e_pt_cvt == ePointConversion.RCtoXY:
                out.append([pt[1], abs(pt[0] - img_y)])
            elif e_pt_cvt == ePointConversion.RCtoCVXY:
                out.append([pt[1], pt[0]])
            elif e_pt_cvt == ePointConversion.CVXYtoXY:
                out.append([pt[0], abs(pt[1] - img_y)])
            elif e_pt_cvt == ePointConversion.CVXYtoRC:
                out.append([pt[1], pt[0]])
            elif e_pt_cvt == ePointConversion.Unchanged:
                pass
            else:
                raise ValueError('Unknown conversion enumeration for function argument e_pt_cvt, ensure the enum ePointConversion is used.')

    if e_out_format == ePointsFormat.ForPolyLine:
        poly_pts = _np.array(out, dtype='int32')
        return poly_pts.reshape((-1, 1, 2))
    elif e_out_format == ePointsFormat.XY:
        return out
    elif e_out_format == ePointsFormat.XXXX_YYYY:
        return list(zip(*out))
    elif e_out_format == ePointsFormat.XYWH:
        x, y = list(zip(*out))
        return (min(x), min(y), max(x) - min(x), max(y) - min(y))
    elif e_out_format == ePointsFormat.RCHW:
        c, r = list(zip(*out))
        return (min(r), min(c), max(r) - min(r), max(c) - min(c))
    else:
        raise ValueError('Unknown output format specified for function argument e_out_format')


def points_normalize(pts, h, w):
    '''(2-list, int|float, int|float)->2-list
    Normalize a list of pts
    pts: A list of pts, i.e. [[0,0],[10,10]]
    h: Image height (rows)
    w: Image width (cols)

    Returns:
        list of coverted points
    '''
    pts_ = list(pts)
    d = _baselib.depth(pts)
    if d == 1:
        pts_ = [pts]
    d = _baselib.depth(pts_)
    assert d == 2, 'Depth of pts should be 1 or 2. Got %s' % d
    out = [[pt[0]/w, pt[1]/h] for pt in pts_]
    return out


def points_denormalize(pts, h, w, asint=True):
    '''(2-list, int|float, int|float)->2-list
    Denormalize a list of pts
    pts: A list of pts, i.e. [[0,0],[10,10]]
    h: Image height (rows)
    w: Image width (cols)

    Returns:
        list of coverted points
    '''
    f = lambda x: int(round(x, 0)) if asint else x
    pts_ = list(pts)
    d = _baselib.depth(pts)
    if d == 1:
        pts_ = [pts]
    d = _baselib.depth(pts_)
    assert d == 2, 'Depth of pts should be 1 or 2. Got %s' % d
    out = [[f(pt[0] * w), f(pt[1] * h)] for pt in pts_]
    return out


def sample_rect(img, w, h):
    '''(str|ndarray,int,int,int,int)->ndarray|None
    Return a retangle of an image as an ndarray
    randomly chosen from the original image.

    ndarray or the path to an image can be used.

    Returns None if the image is smaller than the area
    '''
    # if isinstance(img, str):
    #   img = _cv2.imread(path.normpath(img) , -1)
    img = _getimg(img)
    img_w, img_h = _info.ImageInfo.getsize(img)
    if img_w < w or img_h < h:
        return None

    rnd_col = _randint(0, img_w - w)  # 0 index
    rnd_row = _randint(0, img_h - h)

    I = img[rnd_row:rnd_row + h, rnd_col:rnd_col + w]
    return I


def cropimg_xywh(img, x, y, w, h):
    '''(str|ndarray, int, int, int, int)->ndarray, bool
    Return a rectangular region from an image. Also see transforms.crop.

    Crops to the edge if area would be outside the
    bounds of the image.

    x, y:
        Define the point form which to crop, CVXY assumed
    w, h:
        Size of region

    Returns:
        cropped image area,
        boolean indicating if crop was truncated to border
        of the image

    Notes:
        transforms.crop provides conversion and cropping
        around a point
    '''
    assert isinstance(img, _np.ndarray)
    relu = lambda x: max(0, x)
    crop_truncated = (relu(y), min(y+h, img.shape[0]), relu(x), min(x+w, img.shape[1]))
    crop = (y, y+h, x, x+w)
    return img[relu(y):min(y+h, img.shape[0]), relu(x):min(x+w, img.shape[1])], crop_truncated == crop


def cropimg_pts(img, corners):
    '''(str|ndarray, 4-list|tuple|ndarray) -> ndarray
    Return a rectangular region from an imag. Also see transforms.crop

    corners:
        List of 4 CVXY points of a rectangle.

    Returns:
        The image cropped to the rectangle

    Notes:
        transforms.crop provides conversion and cropping
        around a point
    '''
    assert isinstance(img, _np.ndarray)
    r, c, h, w = rect_as_rchw(corners)
    img_out, _ = cropimg_xywh(img, c, r, w, h)
    return img_out


def poly_area(pts=None, x=None, y=None):
    '''(list|ndarray|None, list|None, list|None) -> float
    Calculate area of a polygon defined by
    its vertices.

    Supports CVXY or XXXXYYYY format

    Example:
        >>>poly_area(x=[1,2,3,4], y=[5,6,7,8])
        >>>poly_area(pts=[(1,5),(2,6),(3,7),(4,8)])
    '''
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    return 0.5 * _np.abs(_np.dot(x, _np.roll(y, 1)) - _np.dot(y, _np.roll(x, 1)))


def centroid(pts):
    '''(ndarray|list) -> 2-tuple
    Calculate the centroid of non-self-intersecting polygon.

    pts:
        Numeric array of points, e.g [[1,2],[10,12], ...]

    Returns:
        2-tuple of the centroid, e.g. (10, 15)
    '''
    return (sum([pt[0] for pt in pts]) / len(pts), sum([pt[1] for pt in pts]) / len(pts))


def roi_polygons_get(img, points):
    '''(ndarray or path, [tuple list|ndarray])->ndarray, ndarray, ndarray, ndarray
    Points are a tuple list e.g. [(0,0), (50,0), (0,50), (50,50)]

    Returns 3 ndarrays and a masked array
    [0] White pixels for the bounding polygon, cropped to a rectangle bounding the roi
    [1] A numpy masked array where pixels outside the roi are masked (False)
    [2] The original image inside the polygon, with black pixels outside the roi
    [3] The original image cropped to a rectangle bounding the roi
    '''

    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    img = _getimg(img)
    white_mask = _np.zeros(img.shape, dtype=_np.uint8)

    roi_corners = _cv2.convexHull(_np.array([points], dtype=_np.int32))
    roi_corners = _np.squeeze(roi_corners)

    if _opencvlib.info.ImageInfo.typeinfo(img) & _opencvlib.info.eImgType.CHANNEL_1.value:
        channel_count = 1
    else:
        channel_count = img.shape[2]  # i.e.  3 or 4 depending on your image

    ignore_mask_color = (255,) * channel_count
    _cv2.fillConvexPoly(white_mask, roi_corners, ignore_mask_color)

    rect = bounding_rect_of_poly(_np.array([points], dtype=_np.int32), as_points=False) #x,y,w,h
    bitwise = _cv2.bitwise_and(img, white_mask)
    rectcrop, _ = cropimg_xywh(bitwise, *rect)
    white_mask_crop, _ = cropimg_xywh(white_mask, *rect)
    mask = _ma.masked_values(white_mask_crop, 0)

    return white_mask_crop, mask, bitwise, rectcrop



def get_image_from_mask(img, mask):
    '''(ndarray, ndarray)->ndarray
    Apply a white mask representing an roi
    to image.

    img and mask must be the same size,
    otherwise None is returned
    '''

    img = _getimg(img)
    if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
        return None

    if len(img.shape) != len(mask.shape):
        if _info.ImageInfo.typeinfo(img) & _info.eImgType.CHANNEL_1.value: #1 channel image, need 1 channel mask
            mask = _cv2.cvtColor(mask, _cv2.COLOR_BGR2GRAY)
        elif _info.ImageInfo.typeinfo(img) & _info.eImgType.CHANNEL_3.value: #3 channel image, need 3 ch mask
            mask = _cv2.cvtColor(mask, _cv2.COLOR_GRAY2BGR)
        elif _info.ImageInfo.typeinfo(img) & _info.eImgType.CHANNEL_4.value:
            mask = _cv2.cvtColor(mask, _cv2.COLOR_GRAY2BGR)
            img = img[:, :, 0:3]
        else:
            assert 1 == 2 #looks like unexpected condition
            return None

    bitwise = _cv2.bitwise_and(img, mask)
    return bitwise


def roi_rescale(roi_pts, proportion=1.0, h=None, w=None):
    '''(ndarray|list|tuple, float, int|None, int|None) -> ndarray
    Grow or shrink an roi around the centre of
    the roi. CVXY is implied.

    roi_pts:
        Array of points [[0,0], [10,10] ....]
    h, w:
        Cap for the width and height, i.e. the roi will be set to h or w
        if it would h or w after rescaling. Lower cap of 0 is also applied.

    Returns:
        Array of rescaled points, eg.
        [[0,0], [10,10] ....]
    '''
    centre = centroid(roi_pts)
    pts = [[_get_limited_val(x, proportion, centre[0], w), _get_limited_val(y, proportion, centre[1], h)] for x, y in roi_pts]
    return pts


def _get_limited_val(v, proportion, centre, limit):
    '''(int|float, float, int|float, int|float|none) -> int
    getval, limted by 0 and limit
    '''
    g = lambda x: proportion * x + (1 - proportion) * centre
    out = int(g(v))
    if limit:
        if out < 0:
            return 0
        if out > limit:
            return limit
    return out


def roi_rescale2(roi_pts, proportion_x=1.0, proportion_y=1.0, h=None, w=None):
    '''(ndarray|list|tuple, float, float, int|None, int|None) -> ndarray
    Grow or shrink an roi around the centre of
    the roi. CVXY is implied.

    roi_pts:
        Array of points [[0,0], [10,10] ....]
    proportion_x, proportion_y:
        Proportion to grow width and height
    h, w:
        Cap for the width and height, i.e. the roi will be set to h or w
        if it would h or w after rescaling. Lower cap of 0 is also applied.
    Returns:
        Array of rescaled points, eg.
        [[0,0], [10,10] ....]
    '''
    centre = centroid(roi_pts)
    pts = [[_get_limited_val(x, proportion_x, centre[0], w), _get_limited_val(y, proportion_y, centre[1], h)] for x, y in roi_pts]
    return pts



def roi_resize(roi_pts, current, target):
    '''(ndarray|list|tuple, 2-tuple, 2-tuple, bool) -> ndarray

    Given an array like of 2d points, transform from the original
    image size to which they refer, to the new image size.

    **Note, this is to maintain the same selection when
    we resize the image, and will not grow the roi**

    roi_points in OpenCV XY frame, where y has origin at image row 0

    roi_pts:
        An array like (which is converted to a numpy array) of points
        in opencv xy format.
    current:
        size of 'image' which points are from, (w,h)
    target:
        size of image to project points on, (w,h)

    returns:
        numpy array of resized points

    comments:
        Use roi.points_convert prior to passing if your points
        are in an RC or cartesian frame.

    '''
    t_mat = _np.eye(2)
    t_mat[0, 0] = target[0]/current[0]
    t_mat[1, 1] = target[1]/current[1]

    return _np.array(_np.matrix(roi_pts) * _np.matrix(t_mat), dtype='uint8')


def roi_rotate(roi_pts, angle, frame_x, frame_y):
    '''(ndarray|list|tuple, float, int, int) -> ndarray
    Rotate an array of 2d points by angle.

    roi_points in OpenCV XY frame, where y has origin at image row 0

    host image size is necessary as rotation occurs around the image center
    and not the origin.

    roi_pts:
        An array like (which is converted to a numpy array) of points
        in opencv xy format.
    angle:
        angle of rotation, negative is clockwise
    frame_x:
        width of image frame
    frame_y:
        height of image frame

    returns:
        numpy array of resized points

    comments:
        Use roi.points_convert prior to passing if your points
        are in an RC or cartesian frame.
    '''
    pts = [_geom.rotate_point(pt, angle, (frame_x, frame_y)) for pt in roi_pts]
    return _np.array(pts)


def pts_reverse(pts):
    '''(ndarray|list|tuple) -> ndarray
    Takes an array of points and reverses it, this
    is effectively an xy to rc or vica versa conversion

    pts:
        listlike array of points

    returns
        ndarray of points, reversed.
    '''
    ndpts = _np.array(pts)
    assert isinstance(ndpts, _np.ndarray)
    assert len(ndpts[0]) == 2, 'Expected points to contain 2 numbers, not %s' % len(ndpts[0]) == 2
    return _np.flip(ndpts, 1)


def to_rect(a):
    '''(arraylike)->ndarray of type float64
    Takes a point [1,2] and returns
    a rectangle sized according to the
    origin [0,0] and the point.

    a:
        single point e.g. (1,2)

    return:
        4-tuple, e.g. (0, 0, 1, 2)

    Example: to_rect([2,3]) would return (0, 0, 2, 3)
    '''
    a = _np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return _np.array(a, _np.float64).reshape(2, 2)


def rect_as_rchw(pts):
    '''(ndarray|list|tuple)-> int, int, int, int
    Take points in CVXY format, and return rectangle defined
    as  r, c, h, w

    pts:
        array of points in CVXY format
    Returns:
        row, col, height, width

    Example:
    >>> pts = [[5,10], [5, 100], [110, 100], [100, 10]]
    >>> rect_as_xywh(pts)
    (5, 10, 106, 91)
    '''
    if len(pts) != 4:
        raise ValueError('Expected 4 points, got %s' % len(pts))

    dims = tuple([len(i) for i in pts])
    if max(dims) != 2 or min(dims) != 2:
        raise ValueError('Some items in iterable argument pts do not have 2 dimensions.')

    pts = _np.array(pts)

    x = pts[:, 0].min()
    y = pts[:, 1].min()
    w = pts[:, 0].max() + 1 - pts[:, 0].min()
    h = pts[:, 1].max() + 1 - pts[:, 1].min()
    return y, x, h, w


# DEBUG bounding_rect_of_poly
def bounding_rect_of_poly(points, as_points=True):
    '''(list|ndarray)->n-list
    Return points of a bounding rectangle in opencv point format if
    as_points=True. Returns integer list.

    Use bounding_rect_of_poly2 if want floats.

    If as_points is false, returns as a tuple (x,y,w,h)

    Returns corner points ([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]
    and *not* top left point with width and height (ie x,y,w,h).
    Note opencv points have origin in top left
    and are (x,y) i.e. col,row (width,height).
    '''
    if not isinstance(points, _np.ndarray):
        points = _np.array([points], dtype=_np.int32)

    #round negatives more negative, positives more positive, and convert to int - boundingrect fails if not integers
    #Note: boundingRect returns Top Left coordinate, it accepts points
    x, y, w, h = _cv2.boundingRect(_np.array(_rnd(points), 'int'))

    if as_points:
        return rect_as_points(y, x, w, h)

    return (x, y, w, h)



# DEBUG bounding_rect_of_ellipse
def bounding_rect_of_ellipse(centre_point, rx, ry):
    '''(list|tuple,int,int)->list
    center_point: x,y  ie. col,row
    Return points of a bounding rectangle in opencv point format.

    For circle, pass in the radius twice.

    Returns corner points ([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]
    and *not* top left point with width and height (ie x,y,w,h).
    Note opencv points have origin in top left
    and are (x,y) i.e. col,row (width,height). Not the matrix standard.
    '''
    x, y = centre_point
    # DEBUG Check outputs of bounding_ellipse_as_points
    return [[x - rx, y - ry], [x + rx, y - ry], [x - rx, y + ry], [x + rx, y + ry]]


def rect2rect_mtx(src, dst):
    '''no idea what this does!'''
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = _np.float64([[cx, 0, tx], [0, cy, ty], [0, 0, 1]])
    return M


def rects_intersect(rect1, rect2):
    '''
    Function to calculate overlapping areas
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the
    # rectangles
    x1_tl = rect1[0]
    x2_tl = rect2[0]
    x1_br = rect1[0] + rect1[3]
    x2_br = rect2[0] + rect2[3]
    y1_tl = rect1[1]
    y2_tl = rect2[1]
    y1_br = rect1[1] + rect1[4]
    y2_br = rect2[1] + rect2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = rect1[3] * rect2[4]
    area_2 = rect2[3] * rect2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def nms_rects(detections, threshold=.5):
    '''
    This function performs Non-Maxima Suppression.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    '''
    if not detections:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
                        reverse=True)
    # Unique detections will be appended to this list
    new_detections = []
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if rects_intersect(detection, new_detection) > threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections


def iou(pts_gt, pts):
    '''(list, list) -> float

    pts_gr, pts:
        list of points, e.g. [[1,0],[0,1], [0,0],[1,1]]
    Return the intersection over union score

    Example:
    >>>iou([[1,0],[0,1], [0,0],[1,1]], [[0.5,0],[0,0.5], [0,0],[0.5,0.5]])
    0.25

    Notes:
        Orders the points prior to calculating
    '''
    pts_gt_ = _geom.order_points(pts_gt)
    pts_ = _geom.order_points(pts)

    x, y = zip(*pts_gt_)
    gt_xmax = max(x)
    gt_xmin = min(x)
    gt_ymax = max(y)
    gt_ymin = min(y)

    x, y = zip(*pts_)
    xmax = max(x)
    xmin = min(x)
    ymax = max(y)
    ymin = min(y)

    dx = min(gt_xmax, xmax) - max(gt_xmin, xmin)
    dy = min(gt_ymax, ymax) - max(gt_ymin, ymin)

    overlap = 0
    if (dx >= 0) and (dy >= 0):
        overlap = dx*dy
    total_area = ((xmax - xmin) * (ymax - ymin)) + ((gt_xmax - gt_xmin) * (gt_ymax - gt_ymin))
    union_area = total_area - overlap
    return overlap / union_area


def iou2(gt_xmin, gt_xmax, gt_ymin, gt_ymax, xmin, xmax, ymin, ymax):
    '''(float, float, float, float, float, float, float, float) -> float

    Args:
        coordinate min and maxes

    Returns the intersection over union score

    Example:
    >>>iou2(0, 1, 0, 1, 0, 0.5, 0, 0.5)
    0.25

    Notes:
        Orders the points prior to calculating
    '''

    if None in [gt_xmin, gt_xmax, gt_ymin, gt_ymax, xmin, xmax, ymin, ymax]:
        return None

    dx = min(gt_xmax, xmax) - max(gt_xmin, xmin)
    dy = min(gt_ymax, ymax) - max(gt_ymin, ymin)

    overlap = 0
    if (dx >= 0) and (dy >= 0):
        overlap = dx*dy
    total_area = ((xmax - xmin) * (ymax - ymin)) + ((gt_xmax - gt_xmin) * (gt_ymax - gt_ymin))
    union_area = total_area - overlap
    return overlap / union_area
