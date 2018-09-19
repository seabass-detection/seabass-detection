# pylint: disable=C0103, too-few-public-methods, locally-disabled
'''Work with aruco markers
'''
from enum import Enum as _Enum

import numpy as _np
import cv2 as _cv2
from sympy.geometry import centroid as _centroid
from sympy.geometry import Point2D as _Point2D


from opencvlib import getimg as _getimg
from opencvlib.common import draw_str as _draw_str
from opencvlib.common import draw_points as _draw_points
import opencvlib.color as _color
from opencvlib.view import pad_images as _pad_images
from opencvlib.geom import order_points as _order_points

_dictionary = _cv2.aruco.getPredefinedDictionary(_cv2.aruco.DICT_ARUCO_ORIGINAL)

MARKERS = {0:'DICT_4X4_50', 1:'DICT_4X4_100', 2:'DICT_4X4_250', \
        3:'DICT_4X4_1000', 4:'DICT_5X5_50', 5:'DICT_5X5_100', 6:'DICT_5X5_250', 7:'DICT_5X5_1000', \
        8:'DICT_6X6_50', 9:'DICT_6X6_100', 10:'DICT_6X6_250', 11:'DICT_6X6_1000', 12:'DICT_7X7_50', \
        13:'DICT_7X7_100', 14:'DICT_7X7_250', 15:'DICT_7X7_1000', 16:'DICT_ARUCO_ORIGINAL'}

DIAGONAL25mm = 35.3553390593274
DIAGONAL30mm = 42.4264068711929
DIAGONAL50mm = 70.7106781186548

class eMarkerID(_Enum):
    '''enum for my sizes'''
    Any_ = -99
    Unknown = -1
    Sz25 = 49
    Sz25_flip = 304
    Sz30 = 18
    Sz30_flip = 528
    Sz50 = 22
    Sz50_flip = 592


class Marker():
    '''represents a single detected marker

    Can be created with points in any order.

    Properties:
        diagonal_length_mm: the geommetrically calculated diagonal length of the marker in mm (e.g. 1.41421 ... for a 1cm square)
        side_length_mm: the side length in mm
        vertices_...: sympy Point2D representations of the corners

    Example:
        M = Marker([[0,0], [10,10], [10,0], [0,10]])
    '''

    def __init__(self, pts, markerid):
        '''init'''
        assert isinstance(pts, (_np.ndarray, list, tuple))
        assert len(pts) == 4, 'Expected 4 points, got %s' % len(pts)
        self.side_length_mm = None
        self.diagonal_length_mm = None
        p = _order_points(pts)
        self.vertices_topleft = _Point2D(p[0][0], p[0][1])
        self.vertices_topright = _Point2D(p[1][0], p[1][1])
        self.vertices_bottomright = _Point2D(p[2][0], p[2][1])
        self.vertices_bottomleft = _Point2D(p[3][0], p[3][1])
        self.markerid = markerid
        self._setid()

    def __repr__(self):
        '''pretty print'''
        info = []
        info.append('Marker "%s"' % self.markerid)
        info.append(str(tuple(self.vertices_topleft) if isinstance(self.vertices_topleft, _Point2D) else ''))
        info.append(str(tuple(self.vertices_topright) if isinstance(self.vertices_topright, _Point2D) else ''))
        info.append(str(tuple(self.vertices_bottomright) if isinstance(self.vertices_bottomright, _Point2D) else ''))
        info.append(str(tuple(self.vertices_bottomleft) if isinstance(self.vertices_bottomleft, _Point2D) else ''))
        return ' '.join(info)

    def _setid(self):
        '''set the markerid from those we expect'''
        if self.markerid in (eMarkerID.Sz25, eMarkerID.Sz25_flip):
            self.side_length_mm = 25.
            self.diagonal_length_mm = DIAGONAL25mm
        elif self.markerid in [eMarkerID.Sz30, eMarkerID.Sz30_flip]:
            self.side_length_mm = 30.
            self.diagonal_length_mm = DIAGONAL30mm
        elif self.markerid in (eMarkerID.Sz50, eMarkerID.Sz50_flip):
            self.side_length_mm = 50.
            self.diagonal_length_mm = DIAGONAL50mm
        else:
            #raise ValueError('Unrecognised markerid "%s". Was the image rotated?' % self.markerid)
            pass


    @property
    def diagonal_px(self):
        '''mean diagonal length'''
        if isinstance(self.vertices_topleft, _Point2D) and isinstance(self.vertices_bottomright, _Point2D):
            x = abs(self.vertices_topleft.distance(self.vertices_bottomright).evalf())
            y = abs(self.vertices_topleft.distance(self.vertices_bottomright).evalf())
            return (x + y)/2
        return None


    @property
    def side_px(self):
        '''mean side length in px'''
        if isinstance(self.vertices_topleft, _Point2D) and isinstance(self.vertices_bottomright, _Point2D):
            a = abs(self.vertices_topleft.distance(self.vertices_topright).evalf())
            b = abs(self.vertices_topleft.distance(self.vertices_bottomleft).evalf())
            c = abs(self.vertices_bottomright.distance(self.vertices_topright).evalf())
            d = abs(self.vertices_bottomright.distance(self.vertices_bottomleft).evalf())
            return (a + b + c + d)/4
        return None


    def px_length_mm(self, use_side=False):
        #DEVNOTE: This has to a method as it uses an argument.
        '''(bool) -> float
        Estimated pixel length in mm, i.e.
        the length of a pixel in mm.

        use_side:
            if true, use the mean side pixel length rather
            than the mean diagonal length
        '''
        if use_side:
            return self.side_length_mm / self.side_px
        return self.diagonal_length_mm / self.diagonal_px

    @property
    def points(self):
        '''Get as list of xy points
        >>>Marker.points
        [[0,10],[10,10],[10,0],[0,0]]
        '''
        return [list(self.vertices_topleft), list(self.vertices_topright), list(self.vertices_bottomright), list(self.vertices_bottomleft)]


    @property
    def centroid(self):
        '''centroid of points'''
        return list(_centroid(self.vertices_bottomleft, self.vertices_bottomright, self.vertices_topleft, self.vertices_topright).evalf())


class Detected():
    '''Detect aruco markers in an image.

    Initalise an instance with an image and then detect
    markers by calling detect on the instance.

    Properties:
        image: The original image
        image_with_detections: The image with detections drawn on it
        Markers: A list containing Marker instances. A Marker instance is a detected marker.

    '''


    def __init__(self, img, detect=True):
        '''(ndarray|str, Enum|List|Tuple, bool)
        '''
        self.Markers = []
        self.image = _getimg(img)
        self.image_with_detections = _np.copy(self.image)
        if detect:
            self.detect()

    def detect(self, expected=eMarkerID.Any_):
        '''(int|tuple|list) -> list

        Detect markers, returning those detected
        as a list of Marker class instances

        expected:
            single eMarkerID value or list of them
            only markers which match will be detected
        '''
        self.Markers = []
        res = _cv2.aruco.detectMarkers(self.image, _dictionary)
        #res[0]: List of ndarrays of detected corners [][0]=topleft [1]=topright [2]=bottomright [3]=bottomleft. each ndarray is shape 1,4,2
        #res[1]: List containing an ndarray of detected MarkerIDs, eg ([[12, 10, 256]]). Shape n, 1
        #res[2]: Rejected Candidates, list of ndarrays, each ndarray is shape 1,4,2

        if not isinstance(expected, (tuple, list, set)):
            expected = [expected]

        if res[0]:
            #print(res[0],res[1],len(res[2]))
            P = _np.array(res[0]).squeeze().astype('int32')

            for ind, markerid in enumerate(res[1]):
                if isinstance(markerid, (tuple, list, _np.ndarray)):
                    markerid = markerid[0]

                if not markerid in [x.value for x in eMarkerID]:
                    continue

                markerid = eMarkerID(markerid)
                if not (eMarkerID.Any_ in expected or markerid in expected):
                    continue

                if len(P.shape) == 2:
                    pts = P
                else:
                    pts = P[ind]
                M = Marker([pts[0], pts[1], pts[2], pts[3]], markerid)
                self.Markers.append(M)

                s = '{0} mm. Px:{1:.2f} mm'.format(int(M.side_length_mm), M.px_length_mm())
                _draw_str(self.image_with_detections, pts[0][0], pts[0][1], s, color=(0, 255, 0), scale=0.6)
                _draw_str(self.image_with_detections, M.centroid[0], M.centroid[1], markerid.name, color=(255, 255, 255), scale=0.7, box_background=(0, 0, 0), centre_box_at_xy=True)
                self.image_with_detections = _draw_points(pts, self.image_with_detections, join=True, line_color=(0, 255, 0), show_labels=False)
                #_cv2.aruco.drawDetectedMarkers(self.image_with_detections, self._results, None, borderColor=(0, 0, 255))
        else:
            self.Markers = []
            self.image_with_detections = _np.copy(self.image)
        return self.Markers


def getmarker(markerid, sz_pixels=500, border_sz=0, border_color=_color.CVColors.white, borderBits=1, saveas=''):
    '''(int, int, int, 3-tuple) -> ndarry
    Get marker image, i.e. the actual marker
    for use in other applications, for printing
    and saving as a jpg.

    markerid:
        Dictionary lookup for MARKERS
    sz_pixels:
        Size of the whole marker in pixels, including the border
    border_sz:
        Border added around the marker, usually white, in pixels.
        This is added after the library has created the marker.
    border_color:
        tuple (0,0,0)
    borderBits:
        the padding around the marker, in image pixels, where an
        image pixel is an single aruco marker building block, a
        marker is a 5 x 5 block. This is added by the library.

    saveas:
        filename to dump img to
    Returns:
        Image as an ndarray
    '''
    m = _cv2.aruco.drawMarker(_dictionary, id=markerid, sidePixels=sz_pixels, borderBits=borderBits)
    m = _cv2.cvtColor(m, _cv2.COLOR_GRAY2BGR)
    if border_sz > 0:
        m = _pad_images(m, pad_color=border_color)

    if saveas:
        _cv2.imwrite(saveas, m)
    return m
