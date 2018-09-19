# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, protected-access, unused-import, too-many-return-statements
'''transforms on an image which return an image'''
from math import sqrt as _sqrt
import os.path as _path
from random import shuffle as _shuffle
from random import uniform as _uniform
from math import floor as _floor
from enum import Enum as _Enum
import logging as _logging
from inspect import getsourcefile as _getsourcefile

import cv2 as _cv2
import numpy as _np
import scipy.stats as _stats
import skimage as _skimage

import funclib.baselib as _baselib
from funclib.iolib import quite
import funclib.iolib as _iolib

import opencvlib.decs as _decs
from opencvlib import getimg as _getimg
from opencvlib import color as _color
from opencvlib.color import BGR2HSV, BGR2RGB, HSVtoGrey, togreyscale
import opencvlib.roi as _roi
import opencvlib.geom as _geom




SILENT = True

_pth = _iolib.get_file_parts2(_path.abspath(_getsourcefile(lambda: 0)))[0]
_LOGPATH = _path.normpath(_path.join(_pth, 'features.py.log'))
_logging.basicConfig(format='%(asctime)s %(message)s', filename=_LOGPATH, filemode='w', level=_logging.DEBUG)



_getval = lambda val_in, out_min, out_max, val_in_max: val_in*(out_max - out_min)*(1/val_in_max) + out_min

def _prints(s, log2file=True):
    '''silent print'''
    if not SILENT:
        print(s)
    if log2file:
        _logging.propagate = False
        _logging.info(s)
        _logging.propagate = True

_prints('Logging to %s' % _LOGPATH)




#from scikit-image
#see http://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.is_low_contrast
import skimage.exposure as _exposure


class eCvt(_Enum):
    '''enum for conversion'''
    uint8_to_1minus1 = 0
    unint8_to_01 = 1


class eChannels(_Enum):
    '''color channel indexes for cv2 img.
    Currently used in chswap
    '''
    B = 0
    G = 1
    R = 2


class eRegionFormat(_Enum):
    '''
    RCHW: (r, c, h, w)
    CVXYWH: (x, y, w, h)
    CVXYXYXYXY: [[x,y], [x,y], [x,y], [x,y]], origin at top left
    XYXYXYXY: [[x,y], [x,y], [x,y], [x,y]], cartesian origin
    HW: (h, w), used for cropping an image at a point
    WH: (w, h), used for cropping an image at a point
    '''
    RCHW = 0
    CVXYWH = 1
    CVXYXYXYXY = 2
    XYXYXYXY = 3
    HW = 4
    WH = 5


#region Handling Transforms in Generators
class Transform():
    ''' class to hold and execute a transform

    Transforms should all take img as the first argument,
    hence we should be able to also store cv2 or other
    functions directly.

    Where we cant store 3rd party lib transforms directly
    we will wrap them in transforms.py

    func: the function
    *args, **kwargs: func arguments
    p: probability of execution, between 0 and 1.
    '''
    def __init__(self, func, *args, p=1, **kwargs):
        '''the function and the arguments to be applied

        p is the probability it will be applied
        '''
        self._args = args
        self._kwargs = kwargs
        self._func = func
        self.p = p if 0 <= p <= 1 else 1
        self.img_transformed = None



    @_decs.decgetimgmethod
    def exectrans(self, img, force=False):
        '''(str|ndarray, bool)->ndarray
        Perform the transform on passed image.
        Returns the transformed image and sets
        to class instance variable img_transformed

        force: force execution of the transform
        '''

        if not img is None:
            img_transformed = self._func(img, *self._args, **self._kwargs)
            if isinstance(img_transformed, _np.ndarray):
                self.img_transformed = img_transformed
            elif isinstance(img_transformed, (list, tuple)):
                self.img_transformed = _baselib.item_from_iterable_by_type(img_transformed, _np.ndarray)
            else:
                raise ValueError('Unexpectedly failed to get ndarray image from transforms.exectrans. Check the transformation function "%s" returns an ndarray.' % self._func.__name__)
            return self.img_transformed

        return None


class Transforms():
    '''
    Queue transforms and apply to image in FIFO order.

    properties:
        img: ndarray, loaded as ndarray if str
        img_transformed: img after applying the queued transforms

    methods:
        add: add a transform to the back of the queue
        shuffle: randomly shuffle the transforms
        executeQueue:   apply the transforms to self.img, or
                        pass in a new image


    example:
    >>> from transforms import Transforms as t
    >>> t1 = t.Transform(t.brightness, p=0.5, value=50)
    >>> t2 = t.Transform(t.gamma, gamma=0.7)
    >>> t3 = t.Transform(t.rotate, angle=90)
    >>> ts = t.Transforms(t1, t2, t3)
    >>> ts.shuffle()
    >>> ts.executeQueue('C:/temp/myimg.jpg')
    >>> cv2.imshow(ts.img_transformed)
    '''
    def __init__(self, *args, img=None):
        '''(str|ndarray, Transform(s))

        Transforms can be queued before
        setting img.
        '''
        self._img = _getimg(img)
        self.img_transformed = None
        self._tQueue = []
        self._tQueue.extend(args)


    def __call__(self, img=None, execute=True):
        '''(str|ndarray) -> void
        Set image if not done previously
        '''
        if not img is None:
            self._img = _getimg(img)

        if execute:
            self.executeQueue()


    def add(self, *args):
        '''(Transform|Transforms) -> void
        Queue a transform or many transforms.

        Example:
        >>>t1 = t.Transform(t.brightness, value=50)
        >>>ts = t.Transforms(t1) #initialise a transforms instance and queue 1 transform
        >>>t2 = t.Transform(t.gamma, gamma=0.7)
        >>>t3 = t.Transform(t.rotate, angle=90)
        >>>ts.add(t2, t3) #add 2 more transforms to the queue
        '''
        s = 'Queued transforms ' + ' '.join([f._func.__name__ for f in args])
        _logging.info(s)
        self._tQueue.extend(args)

    def shuffle(self):
        '''inplace random shuffle of the transform queue
        '''
        _shuffle(self._tQueue)


    def executeQueue(self, img=None, print_debug=False):
        '''(str|ndarray)->ndarray
        Execute transformation, FIFO order and
        set img_transformed property to the transformed
        image. Also returns the transformed image.

        img:
            Image file path or ndarray of image.
        print_debug:
            prints the transforms to console.

        Returns:
            transformed image as ndarray
        '''
        if not img is None:
            self._img = _getimg(img)

        first = True
        if _baselib.isempty(self._tQueue):
            return self._img

        for T in self._tQueue:
            pp = _uniform(0, T.p)
            if T.p < pp:
                if print_debug:
                    print('Skipped %s [%s, %s]. (%.2f < %.2f)' % (T._func.__name__, T._args, T._kwargs, T.p, pp))
                break

            if print_debug:
                print('Executing %s [%s, %s]' % (T._func.__name__, T._args, T._kwargs))

            assert isinstance(T, Transform)
            if first:
                self.img_transformed = T.exectrans(self._img)
                first = False
            else:
                self.img_transformed = T.exectrans(self.img_transformed)

        return self.img_transformed
#endregion


@_decs.decgetimgsk
def log(img, gain=1, inv=False):
    '''(ndarray|str, float, bool) -> BGR-ndarray
    '''
    i = _exposure.adjust_log(img, gain=gain, inv=inv)
    assert str(i.dtype) == 'uint8'
    return _color.RGB2BGR(i)


def int32_to_uint8(ndarray, absolute=True):
    '''(ndarray)->ndarray
    Convert array to uint8 if it is int32

    abs:
        perform an abs if True

    return
        converted array
    '''
    assert isinstance(ndarray, _np.ndarray)
    if ndarray.dtype == 'int32':
        return absolute(ndarray).astype('uint8') if absolute else ndarray.astype('uint8')

    return ndarray


@_decs.decgetimgsk
def sigmoid(img, cutoff=0.5, gain=10, inv=False):
    '''(ndarray|str, float, float, bool) -> BGR-ndarray
    Performs Sigmoid Correction on the input image.
    '''
    i = _exposure.adjust_sigmoid(img, cutoff=cutoff, gain=gain, inv=inv)
    return _color.RGB2BGR(i)


@_decs.decgetimgsk
def equalize_adapthist(img, kernel_size=None, clip_limit=0.01, nbins=256):
    '''(ndarray|str, int|listlike, float, int) -> BGR-ndarray
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Supports color.

    kernel_size: integer or list-like, optional
        Defines the shape of contextual regions used in the algorithm.
        If iterable is passed, it must have the same number
        of elements as image.ndim (without color channel).
        If integer, it is broadcasted to each image dimension.
        By default, kernel_size is 1/8 of image height by 1/8 of its width.
    clip_limit : float, optional
        Clipping limit, normalized between 0 and 1 (higher values give more contrast).

    nbins : int, optional
        Number of gray bins for histogram (“data range”).
    '''
    i = _exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)  #i will be floating point, so can't be shown with cv2, unless cvt using np.array(i*255, dtype=("uint8"))
    return _color.RGB2BGR(i) #this func is wrapped to handle black and white as well


def to8bpp(img):
    '''(ndarray:float)->ndarray:uint8
    Convert float image representation to
    8 bit image.
    '''
    assert isinstance(img, _np.ndarray)
    if 'float' in str(img.dtype):
        return _np.array(img * 255, dtype=_np.uint8)
    elif str(img.dtype) == 'uint8':
        return img

    assert img.dtype == 'uint8' #unexpected, debug if occurs
    return img


def toFloat(img, conv=eCvt.unint8_to_01):
    '''(ndarray:uint8)->ndarray:float
    Convert 8bpp image representation to float.

    img:
        ndarray representation of image, unint8
    conv:
        Enumeration eCvt, output can be scaled
        between 0 and 1 or -1 to +1

    Note that if an ndarray of type float is passed in
    then it will be tested for negative values and
    rescaled accordingly.
    '''
    assert isinstance(img, _np.ndarray)
    if 'uint' in str(img.dtype):
        if conv == eCvt.uint8_to_1minus1:
            return  (_np.array(img / 255, dtype=_np.float) - 0.5) * 2
        elif conv == eCvt.unint8_to_01:
            return _np.array(img / 255, dtype=_np.float)
    elif 'float' in str(img.dtype):
        if _np.any(img < 0): #have negatives
            if conv == eCvt.uint8_to_1minus1: #already negatives, return unchanged
                return img
            elif conv == eCvt.unint8_to_01:
                return (img - 0.5) * 2
            else:
                raise ValueError('Unexpected array datatype passed to transforms.toFloat()')
        else: #no negatives
            if conv == eCvt.unint8_to_01: #no negatives and a float, return the float
                return img
            elif conv == eCvt.uint8_to_1minus1:
                return (img - 0.5) * 2
            else:
                raise ValueError('Unexpected array datatype passed to transforms.toFloat()')



    assert 'float' in str(img.dtype) #unexpected, debug if occurs
    return img


@_decs.decgetimgsk
def equalize_hist(img, nbins=256, mask=None):
    '''(ndarray|str, int, ndarray of bools, 0 or 1s) -> BGR-ndarray
    Perform historgram equalization with optional mask.
    True mask values only are evaluated
    '''
    i = _exposure.equalize_hist(img, nbins, mask)
    return _color.RGB2BGR(i)


@_decs.decgetimgsk
def intensity(img, in_range='image', out_range='dtype'):
    '''(ndarray|str, str|2-tuple, str|2-tuple) -> ndarray
    Rescales image range to out_range.

    in_range, out_range : str or 2-tuple
    Min and max intensity values of input and output image. The possible values for this parameter are enumerated below.

    ‘image’
        Use image min/max as the intensity range.
    ‘dtype’
        Use min/max of the image’s dtype as the intensity range.
    dtype-name
        Use intensity range based on desired dtype. Must be valid key in DTYPE_RANGE.
    2-tuple
        Use range_values as explicit min/max intensities, in_range and out_range
        are uint8.
        eg (0, 255) or (10, 200)

    Example:
        >>>I = intensity(img, 'image', out_range=(10, 200)) #decrease contrast
        >>>I = intensity(img, (10, 210), out_range=(0, 255)) #increase contrast
    '''
    i = _exposure.rescale_intensity(img, in_range=in_range, out_range=out_range)
    return _color.RGB2BGR(i)
#endregion



def intensity_wrapper(img, intensity_=0):
    '''(ndarray|str, float) -> ndarray
    Friendly wrapper for intensity, allowing
    intensity to be set with a single value.

    intensity_
        Takes a single value between -1 and 1,
        0 is no change, < 0 decreases contrast
        > 0 increases contrast. Sensible values
        are +/-0.5
    '''
    img = _getimg(img)
    assert isinstance(img, _np.ndarray)
    range_ = (img.min(), img.max())

    if -1 > intensity_ > 1:
        raise ValueError('Invalid value for intensity_, intensity not -1 <= intensity <= 1')

    if intensity_ == 0:
        return img

    mid_in = (range_[1] - range_[0]) / 2 #centre point of range, ie (100, 200) = 150
    mid_in_size = abs(range_[1] - mid_in) #1/2 the size of range, ie 50
    mid_out_size = mid_in_size

    if intensity_ < 0: #intensity decrease, shrink output range, increase input range
        in_range = (0, 255)
        sz = _floor(_getval(abs(intensity_), 0, mid_out_size, 1))
        out_range = (range_[0] + sz, range_[1] - sz)
    else: #intensity increase, shrink input range, leave output range untouched
        out_range = (0, 255)
        sz = _floor(_getval(abs(intensity_), 0, mid_in_size, 1))
        in_range = (range_[0] + sz, range_[1] - sz)

    i = intensity(img, in_range=in_range, out_range=out_range)
    return i



def chswap(img, new_order):
    '''(str|ndarray, tuple|list) -> ndarray
    Change the channel order

    new_order:
        new channel order as a tuple, so
        (0,2,1) whould swap ch2 to ch1 etc.

    Comment:
        Can also be used to convert any single
        channel to grayscale.

    Examples:
    Swap BGR to RGB
    chswap(img, (2,1,0)

    '''

    img = _getimg(img)
    i = _np.dsplit(img, 3)
    return _np.dstack((i[new_order[0]], i[new_order[1]], i[new_order[2]]))


def crop(img, region, eRegfmt=eRegionFormat.RCHW, around_point=None, allow_crop_truncate=True):
    '''(ndarray, list|tuple, Enum:roi.eRegionFormat, 2-tuple|None, bool) -> ndarray
    Crops an image.

    img:
        The image
    region:
        Coordinate array, a 4-tuple/list in format defined by ePtType
        Tuple can be a 1-deep list if in rchw like format,
        or a 4-tuple of points, e.g. ((0, 0), (100, 100), (0, 100), (100,0))
        If around_point is True, then the region must be just wh, or hw
    eRegfmt:
        The format of the points in variable region
    around_point:
        If none, standard crop assuming some xywh 4-tuple passed,
        otherwise around_point is a CVXY 2-tuple point and array is
        in WH or HW format (i.e. a 2-tuple).
    allow_crop_truncate:
        If true, will crop to the image edges if the region covers an
        area outside the image, otherwise an error is raised

    Returns:
        The image

    Examples:
    >>>

    '''
    assert isinstance(img, _np.ndarray)
    if around_point:
        assert eRegfmt == eRegionFormat.WH or eRegfmt == eRegionFormat.HW, \
            'Cropping was requested to be around a point, but the RegionFormat was not eRegionFormat.WH or eRegionFormat.HW'
        assert len(region) == 2, 'Cropping around a point, expected region to be a 2-tuple but got a %s tuple' % len(region)
    else:
        assert len(region) == 4, 'Cropping with rchw like area, expected region to be a 4-tuple but got a %s tuple' % len(region)

    r = 0; c = 0; w = 0; h = 0

    if around_point: #around_point is a 2-tuple CVXY point
        if eRegfmt == eRegionFormat.HW:
            w = region[1]; h = region[0]
        elif eRegfmt == eRegionFormat.WH:
            w = region[0]; h = region[1]
        else:
            raise ValueError('around_points was provided but an invalid eRegfmt argument was passed. eRegfmt should be WH or HW')
        c = around_point[0] - int(w/2)
        r = around_point[1] - int(h/2)
    else:
        if eRegfmt == eRegionFormat.CVXYWH:
            r = region[1]; h = region[3]; c = region[0]; w = region[2]
        elif eRegfmt == eRegionFormat.CVXYXYXYXY:
            r, c, h, w = _roi.rect_as_rchw(region)
        elif eRegfmt == eRegionFormat.RCHW:
            r = region[0]; h = region[2]; c = region[1]; w = region[3]
        elif eRegfmt == eRegionFormat.XYXYXYXY:
            pts = _roi.points_convert(region, img.shape[1], img.shape[0], _roi.ePointConversion.XYtoCVXY, _roi.ePointsFormat.XY)
            r, c, h, w = _roi.rect_as_rchw(pts)
        else:
            raise ValueError('Unknown region format enumeration in argument eRegfmt')

    fixy = lambda y: max([min([y, img.shape[0]]), 0])
    fixx = lambda x: max([min([x, img.shape[1]]), 0])

    if allow_crop_truncate:
        pts = [[fixx(c), fixy(r)], [fixx(c + w), fixy(r)], [fixx(c), fixy(r + h)], [fixx(c + w), fixy(r + h)]]
    else:
        pts = [[c, r], [c + w, r], [c, r + h], [c + w, r + h]]
        pts_xxxyyy = list(zip(*pts)) #[[1,2],[3,4] -> [[1, 3], [2, 4]]
        fpts = _baselib.list_flatten(pts)
        if min(fpts) < 0 \
                or max(pts_xxxyyy[0]) + 1 > img.shape[1] \
                or max(pts_xxxyyy[1]) + 1 > img.shape[0]:
            raise ValueError('allow_crop_truncate was false, but the crop area was out of the bounds of the image shape')

    r, c, h, w = _roi.rect_as_rchw(pts)
    h -= 1; w -= 1
    i, _ = _roi.cropimg_xywh(img, c, r, w, h)
    return i


def resize(image, width=None, height=None, inter=_cv2.INTER_AREA):
    '''(ndarray|str, int, int, constant)->ndarray
    Resize an image, to width or height, maintaining the aspect.

    image:
        an image or path to an image
    width:
        width of image
    height:
        height of image
    inter:
        interpolation method

    Returns:
        An image

    Notes:
        Returns original image if width and height are None.
        If width or height or provided then the image is resized
        to width or height and the aspect ratio is kept.
    '''
    image = _getimg(image)
    dim = None
    (h, w) = image.shape[:2]
    image = _getimg(image)
    if width is None and height is None:
        return image
    elif width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    dim = (int(dim[0]), int(dim[1]))
    return _cv2.resize(image, dim, interpolation=inter)


def rotate(image, angle, no_crop=True):
    '''(str|ndarray, float, bool) -> ndarray
    Rotate an image through 'angle' degrees.

    image:
        the image as a path or ndarray
    angle:
        angle, positive for anticlockwise, negative for clockwise
    no_crop:
        if true, the image will not be cropped

    '''
    img = _getimg(image)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = _cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    if no_crop:
        cos = _np.abs(M[0, 0])
        sin = _np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return _cv2.warpAffine(img, M, (nW, nH))

    return _cv2.warpAffine(img, M, (w, h))


def rotate2(image, angle, no_crop=True):
    '''(str|ndarray, float, bool) -> ndarray, list
    Rotate an image through 'angle' degrees.

    The x-y translation represents the translation
    of each point in the opencv frame due to the
    change in image size. The xy translation can
    be passed to geom.rotate_points if rotate2
    occured with no_crop=True.

    image:
        the image as a path or ndarray
    angle:
        angle, positive for anticlockwise, negative for clockwise
    no_crop:
        if true, the image will not be cropped

    Returns:
        rotated image, x-y translation
    '''
    img = _getimg(image)
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = _cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    if no_crop:
        cos = _np.abs(M[0, 0])
        sin = _np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        xtrans = (nW / 2) - cX
        ytrans = (nH / 2) - cY
        # adjust the rotation matrix to take into account translation
        M[0, 2] += xtrans
        M[1, 2] += ytrans

        img_rot = _cv2.warpAffine(img, M, (nW, nH))
        return img_rot, (xtrans, ytrans)

    img_rot = _cv2.warpAffine(img, M, (w, h))
    return img_rot, (0, 0)


def gamma(img, gamma_=1.0):
    '''(str|ndarray, float) -> ndarray
    Adjust gamma of an image

    gamma:
        1 means no adjustment, range for gamma_
        is 0 -> infinity
        Sensible ranges are 0 -> 5
    '''
    img = _getimg(img)
    invGamma = 1 if gamma_ == 0 else 1.0 / gamma_
    table = _np.array([((i / 255.0) ** invGamma) * 255 for i in _np.arange(0, 256)]).astype("uint8")
    itmp = _cv2.LUT(img, table)
    return itmp


#skimage transforms in skimage.exposure
@_decs.decgetimgsk
def gamma1(img, gamma_=1, gain=1):
    '''(ndarray|str, float, float) -> BGR-ndarray
    Uses skimage version.

    Performs Gamma Correction on the input image.
    Transforms the input image pixelwise according to the equation O = I**gamma after scaling each pixel to the range 0 to 1.

    eg: gamma_corrected = exposure.adjust_gamma(image, 2)
    '''
    i = _exposure.adjust_gamma(img, gamma_, gain)
    return _color.RGB2BGR(i)


def denoise(img, sigma):
    '''uses the dctDenoising algo from
    opencv to denoise an image.

    sigma:
        value, typically between 0 and 50.
    '''
    i = _getimg(img)
    iout = _np.copy(i)
    _cv2.xphoto.dctDenoising(i, iout, sigma)
    return iout


def brightness(img, value):
    '''(ndarray|str, int) -> ndarray
    Adjust brightness of image.

    Simply does a clipped add of value to the v channel
    after converting image to HSV.
    '''
    img = _getimg(img)

    if value == 0:
        return(img)

    hsv = _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)
    h, s, v = _cv2.split(hsv)
    value = int(value)
    lim_upper = 255 - value
    v[v > lim_upper] = 255

    lim_lower = abs(value)
    v[v < lim_lower] = 0

    v[_np.bitwise_and(v > lim_lower, v <= lim_upper) is True] += _np.uint8(value)


    final_hsv = _cv2.merge((h, s, v))
    img = _cv2.cvtColor(final_hsv, _cv2.COLOR_HSV2BGR)
    return img


def histeq_color(img, cvtToHSV=True):
    '''(ndarray)->ndarray
        Equalize histogram of color image
        '''
    img = _getimg(img)

    if cvtToHSV:
        img_yuv = _cv2.cvtColor(img, _cv2.COLOR_BGR2YUV)
    else:
        img_yuv = img

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = _cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return _cv2.cvtColor(img_yuv, _cv2.COLOR_YUV2BGR)


@_decs.decgetimg
def histeq_adapt(img, clip_limit=2, tile_size=(8, 8)):
    '''(ndarray|str, int, (int,int))->ndarray
    Adaptive histogram equalization.
    Performs equaliation in equal size tiles as specified by tile_size.

    tile_size of 8x8 will divide the image into 8 by 8 tiles

    clip_limit set the threshold for contrast limiting.

    img is converted to black and white, as required by cv2.createCLAHE
    '''
    img_bw = _color.togreyscale(img)
    clahe = _cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(img_bw)


@_decs.decgetimg
def histeq(im):
    '''(ndarray|str, int)->ndarray
    Histogram equalization of a grayscale image.
    '''
    img_bw = _color.togreyscale(im)
    return _cv2.equalizeHist(img_bw)


def compute_average(imlist, silent=True):
    """(list,[bool])->ndarray
        Compute the average of a list of images. """

    # open first image and make into array of type float
    averageim = _np.array(_cv2.imread(imlist[0], -1), 'f')

    skipped = 0

    for imname in imlist[1:]:
        try:
            averageim += _np.array(_cv2.imread(imname), -1)
        except Exception as e:
            if not silent:
                print(imname + "...skipped. The error was %s." % str(e))
                skipped += 1

    averageim /= (len(imlist) - skipped)
    if not silent:
        print('Skipped %s images of %s' % (skipped, len(imlist)))
    return _np.array(averageim, 'uint8')


@_decs.decgetimg
def compute_average2(img, imlist, silent=True):
    """(list,[bool])->ndarray
        Compute the average of a list of images.
        This exists to be compatible with imgpipes transformation framework"""

    # open first image and make into array of type float
    assert isinstance(imlist, list)

    imlist.append(img)
    averageim = _np.array(_cv2.imread(imlist[0], -1), 'f')

    skipped = 0

    for imname in imlist[1:]:
        try:
            averageim += _np.array(_cv2.imread(imname), -1)
        except Exception as e:
            if not silent:
                print(imname + "...skipped. The error was %s." % str(e))
                skipped += 1

    averageim /= (len(imlist) - skipped)
    if not silent:
        print('Skipped %s images of %s' % (skipped, len(imlist)))
    return _np.array(averageim, 'uint8')


@_decs.decgetimg
def sharpen(img):
    '''(ndarray|str, float) -> ndarray
    Sharpen an image

    img:
        An image or file path
    factor:
        scale sharpening
    Returns:
        image
    '''
    image = _getimg(img)
    kernel = _np.array([
                    [-1, -1, -1, -1, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, 2, 8, 2, -1],
                    [-1, 2, 2, 2, -1],
                    [-1, -1, -1, -1, -1]
                    ])
    kernel = kernel/_np.sum(kernel)
    i = _cv2.filter2D(image, -1, kernel)
    return i



def similiarity_matrices(A, B, filter_invalid_pairs=True):
    '''(ndarray|list|tuple, ndarray|list|tuple, bool) -> ndarray
    Given to lists of points, get
    the rotation and translation matrix

    filter_invalid_pairs:
        Will remove point pairs where any value is None/Nan


    Returns the translation matrix compatible with cv2.warpaffine
    for euclidean (translation, rotation) transforms.

    To use the outputs, the image should be rotated around the centre
    point of the target image

    **USE SKIMAGE.TRANSFORMS - NOT THIS. NOW REDUNDANT
    '''
    assert len(A) == len(B)


    N = A.shape[0] # total points
    A_ = _np.asarray(A).copy()
    B_ = _np.asarray(B).copy()

    if filter_invalid_pairs:
        A_, B_ = _geom.valid_point_pairs(A_, B_)

    centroid_A = _np.mean(A_, axis=0)
    centroid_B = _np.mean(B_, axis=0)

    # centre the points
    AA = A_ - _np.tile(centroid_A, (N, 1)) #difference between points and the centroid, i.e. bring to origin
    BB = B_ - _np.tile(centroid_B, (N, 1)) #difference between points and the centroid

    # dot is matrix multiplication for array
    H = _np.dot(_np.transpose(AA), BB)

    V, S, W = _np.linalg.svd(H)
    d = (_np.linalg.det(V) * _np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = _np.dot(V, W).T
    U[:, :-1] = _np.array([centroid_A - centroid_B]).reshape(2, 1)
    return U


def homotrans(H, x, y):
    '''(3x3 ndarray,float,float)->float,float
    return homogenous coordinates
    '''
    xs = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    ys = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    s = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return xs / s, ys / s


def sharpen_unsharpmask(img, kernel_size=(5, 5), threshhold=0.0, weight=1, sigma=3):
    #see pg 185 digital image processing
    '''(ndarray|str, 2-tuple) -> ndarray
    Sharpen an image using unsharp mask, rescales intensity.

    img:
        image or filepath to an image
    kernel_size:
        Size of the guassian kernel, e.g. (9, 9)
    threshhold:
        The mask is only applied where it is above threshhold, after
        the weighting is applied
    weight:
        weight applied to the mask

    Notes:
        beta < 1: Deemphasises contribution of the max
        beta = 1: default sharpening
        beta > 1: highboost filtering, increases sharpness
    '''
    if kernel_size[0]%2 == 0 or kernel_size[1]%2 == 0:
        raise ValueError('Kernel size must be odd, size was (%s, %s)' % (kernel_size[0], kernel_size[1]))

    img = _getimg(img)
    img_float = _skimage.img_as_float(img)
    gb = _cv2.GaussianBlur(img, kernel_size, sigmaX=sigma, sigmaY=sigma)
    gb = _skimage.img_as_float(gb)

    mask = (img_float - gb)*weight
    mask[mask < threshhold] = 0
    usm_img = img_float + mask
    usm_img = _skimage.exposure.rescale_intensity(usm_img, (_np.min(img_float), _np.max(img_float)))
    usm_img = _skimage.img_as_ubyte(usm_img)
    return usm_img


def bilateral_filter(img):
    '''to implement'''
    #TODO Implement
    #http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    #in opencv as cv2.bilateralfilter
    pass
    #Todo add bilatal filtering support
    #http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html
    #https://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#sobel
