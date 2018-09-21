# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
'''deal with colors
See test_color.py for examples of using the filter
'''
from enum import Enum as _Enum

import numpy as _np
import cv2 as _cv2

import funclib.baselib as _baselib

import opencvlib.decs as _decs
import opencvlib.info as _info
from opencvlib import getimg as _getimg


_JET_DATA = {'red': ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1),
                     (1, 0.5, 0.5)),
             'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1),
                       (0.91, 0, 0), (1, 0, 0)),
             'blue': ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0),
                      (1, 0, 0))}

_CMAP_DATA = {'jet': _JET_DATA}



class CVColors():
    '''BGR base color tuples'''
    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)
    light_grey = (200, 200, 200)
    grey = (130, 130, 130)
    dark_grey = (75, 75, 75)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    cyan = (255, 255, 0)
    black = (0, 0, 0)
    white = (255, 255, 255)


class eColorSpace(_Enum):
    '''colorspace types'''
    HSV = 0
    RGB = 1
    BGR = 2
    Grey = 3
    HSV360100100 = 4
    HSV255255255 = 5


class ColorInterval():
    '''class to store a filter range
    in defined color space and
    convert between them.
    
    Initialise with tuples where tuples are paired to represent a range
    e.g. start=(0,0,0), finish=(255,255,255) is then entire range

    Note OpenCV HSV is 0-179,0-255,0-255
    '''
    #This could be done with a LookUpTable approach, this
    #may be more efficient when working around the red hue
    #axis
    BGR_RED = _np.array([[17, 15, 100], [50, 56, 200]]).astype('uint8')
    BGR_BLUE = _np.array([[86, 31, 4], [220, 88, 50]]).astype('uint8')
    BGR_GREEN = _np.array([[103, 86, 65], [145, 133, 128]]).astype('uint8')

    def __init__(self, color_space=eColorSpace.BGR, start=(None, None, None), finish=(None, None, None)):
        self._color_space = color_space
        
        if color_space == eColorSpace.Grey:    
            self._check_intervals()
            if len(start) != 1 and len(finish) != 1:
                raise UserWarning('Color space set to grey, but tuple start and/or finish length greater not equal 1')
        else:
            start = hsvtrans(start, color_space)
            finish = hsvtrans(finish, color_space)

            if 'HSV' in self.color_space.name:
                self._color_space = eColorSpace.HSV

            self._asnumpyinterval = _np.array([start, finish]).astype('uint8')
            self._check_intervals() #check intervals only expects opencv color ranges, so do after any conversions


    
    @property
    def color_space(self):
        '''color_space getter'''
        return self._color_space
    @color_space.setter
    def color_space(self, color_space):
        '''color_space setter'''
        self._asnumpyinterval = self._cvt(color_space)
        self._color_space = color_space
        


    def _check_intervals(self):
        fRGB = lambda x: x if max(x) <= 255 and min(x) >= 0 else None
        fH = lambda x: x if max(x) <= 179 and min(x) >= 0 else None
        fSV = lambda x: x if max(x) <= 255 and min(x) >= 0 else None

        if self.color_space != eColorSpace.HSV:
            if fRGB(self.lower_interval()) is None:
                raise UserWarning('Start range was invalid')

            if fRGB(self.upper_interval()) is None:
                raise UserWarning('Finish range was invalid')
        else:
            if fH(self.lower_interval()[0:1]) is None:
                raise UserWarning('Start range was invalid, hue value should not exceed 179')

            if fSV(self.lower_interval()[1:3]) is None:
                raise UserWarning('Start range was invalid for saturation and brightness')

            if fH(self.upper_interval()[0:1]) is None:
                raise UserWarning('Finish range was invalid, hue value should not exceed 179')

            if fSV(self.upper_interval()[1:3]) is None:
                raise UserWarning('Finish range was invalid for saturation and brightness')
                            

    def lower_interval(self):
        '''return the lbound numpy array
        eg [[0,0,0]]
        '''
        return self._asnumpyinterval[0]


    def upper_interval(self):
        '''return the lbound numpy array
        eg [[0,0,0]]
        '''
        return self._asnumpyinterval[1]
    
     
    def _cvt(self, to):
        '''(Enum:eColorSpace) -> ndarray
        Given a range conversion converts the
        start and finish set on initialisation
        to defined space and returns the ndarray
        as a single range.

        to:
            target colour interval format
        Return:
            Single ndarray of shape 2,3.
            e.g. ([[0,0,0], [255,255,255]])

        '''
        assert isinstance(to, eColorSpace)

        #convert are ranges to pixels for to
        #exploit opencv conversion
        if self._asnumpyinterval.shape[1] >= 3:
            tmp = _np.reshape(self._asnumpyinterval.copy(), [1, 2, 3])
        else: #grey
            tmp = _np.reshape(self._asnumpyinterval.copy(), [1, 2])
            

        tmp = cvt(tmp, self._color_space, to)
        if tmp.shape == (1, 2, 3):
            tmp = _np.reshape(tmp, [2, 3])
        elif tmp.shape == (1, 2):
            tmp = _np.reshape(tmp, [2, 1])
        else:
            raise UserWarning('Could not reshape color interval of shape {!s}'.format(tmp.shape))
        
        return tmp
        



class ColorDetection():
    '''color detection stuff
    Simple example:
        #define a colorinterval in the imagej HSV color space
        ciH = color.ColorInterval(color.eColorSpace.HSV255255255, (33, 0, 0), (255, 255, 102))
        #Create class instance with image and the color interval. img_in is in BGR, we tell opencv to convert to HSV
        CD = color.ColorDetection(img_in, ciH, color.eColorSpace.HSV, no_conversion=False)
        #perform the detection
        CD.detect()
        #This contains the bgr version of the image. Pixels of colors not in colorinterval in the image are set to 0, 0, 0
        return CD.detected_as_bgr()
    '''

    def __init__(self, img, ColInt, color_space=eColorSpace.HSV, no_conversion=False):
        '''(ndarray|str, class:ColorInterval|list:class:ColorInterval, Enum:eColorSpace) -> void

        img:
            Will load img if a path is passed.
        color_space:
           Convert to this colorspace, assumes OpenCV format (BGR) of img
        ColInt:
            Class ColorInterval or list like of ColorInterval instances.
            Colorinterval class specifies an color interval and the
            format of that interval (eg, RGB, BGR, HSV).
            If color_space is Grey, then the image will be converted to greyscale  
        no_conversion:
            Use if img is already in the color_space format, no conversion will occur

        Notes:
            BGR is the assumed format,
            which is converted according to color_space.

            Detection will use the supplied ColorInterval class.
                      
        '''
        img = _getimg(img)
        if not no_conversion:
            img = cvt(img, from_=eColorSpace.BGR, to=color_space) 

        self._color_space = color_space
        self._ColInt = ColInt
        self._img = img

        #self.boolmask = None
        self.img_detected = None


    def detect(self):
        '''(void) -> void
        Carry out the detection after class
        initialisation with 1 or more ColorIntervals

        Returns:
            Nothing returned, self.img_detected is set
            to the detected image
        '''
        if isinstance(self._ColInt, ColorInterval):
            intervals = [self._ColInt]
        else:
            intervals = self._ColInt

        if False in [self._color_space == ci.color_space for ci in intervals]:
            raise UserWarning('A ColorInterval color space {!s} did not match' \
               ' that set in the ColorDetection instance.'.format(self._color_space.name))

        I = self._img.copy()
        m = _np.ndarray([])
        first = True
        for ci in intervals:
            assert isinstance(ci, ColorInterval)
            if first:
                m = _cv2.inRange(I, ci.lower_interval(), ci.upper_interval())
                first = False
            else:
                m = _cv2.bitwise_or(m, _cv2.inRange(I, ci.lower_interval(), ci.upper_interval()))
            
        I = _cv2.bitwise_and(I, I, mask=m) 
        self.img_detected = I    
        #self.boolmask = m.astype('bool')

    
    def detected_as_bgr(self):
        '''(void) ->ndarray|None
        Get the detected image in
        native OpenCV [BGR] format
        '''
        if self.img_detected.size:
            return cvt(self.img_detected, self._color_space, eColorSpace.BGR)

        return None


def hsvtrans(color_in, format_in):
    '''(tuple|list, Enum:eColorSpace)-> 3-tuple
    Convert different HSV scales to the OpenCV HSV scale

    color_in:
        ndarray, eg [340, 90, 90]
    formatin:
        the eColorSpace enumeration,
    HSV is expressed in multiple scales.
    OpenCV: 0-179, 0-255, 0-255
    Internet:0-360,0-100,0-100
    ImageJ:0:255, 0:255, 0:255

    Return:
        HSV in opencv scale

    If format_in is not eColorSpace.HSV255255255 or HSV360100100
    returns color_in without change
    '''
    assert isinstance(format_in, eColorSpace)
    perc_2_bit8 = lambda x: int(x*255/100)
    d360_2_d179 = lambda x: int(x*179/360)
    bit8_2_d179 = lambda x: int(x*179/255)

    if format_in == eColorSpace.HSV255255255: #opencv
        return ([bit8_2_d179(color_in[0]), color_in[1], color_in[2]])
    elif format_in == eColorSpace.HSV360100100: #online pickers
        return (d360_2_d179(color_in[0]), perc_2_bit8(color_in[1]), perc_2_bit8(color_in[2]))

    return color_in


@_decs.decgetimg8bpp
def BGR2RGB(img):
    '''(ndarray)->ndarray
    BGR  to RGB
    opencv to skimage
    '''
    if _info.ImageInfo.isbw(img):
        return img

    return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)


@_decs.decgetimg8bpp
def RGB2BGR(img):
    '''(ndarray)->ndarray
    RGB  to BGR
    skimage to opencv
    '''
    if _info.ImageInfo.isbw(img):
        return img

    return _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)


@_decs.decgetimg
def BGR2HSV(img):
    '''(ndarray)->ndarray
    BGR to HSV
    179,255,255
    '''
    if _info.ImageInfo.isbw(img):
        return img
    return _cv2.cvtColor(img, _cv2.COLOR_BGR2HSV)


@_decs.decgettruegrey
def togreyscale(img):
    '''(str|ndarray)->ndarray
    Convert image to greyscale, assumes BGR
    '''
    return img


def HSVtoGrey(img):
    '''(ndarray) -> ndarray
    Convert hsv to grey.

    This will give different results from
    converting HSV
    '''
    return img[:, :, 2:3].squeeze()


def cvt(img, from_=eColorSpace.BGR, to=eColorSpace.RGB):
    '''(ndarray|list:ndarray, Enum:eColorSpace)
    Convert images'''
    
    f = lambda x: x[0] if _baselib.isIterable(x) and x else x

    if not _baselib.isIterable(img):
        img = [img]
    
    out = []
    for I in img:
        if from_ == eColorSpace.BGR:
            if to == eColorSpace.Grey:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_BGR2GRAY))
            elif to == eColorSpace.HSV:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_BGR2HSV))
            elif to == eColorSpace.RGB:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_BGR2RGB)) 
            elif to == eColorSpace.BGR:
                out.append(I)
            else:
                raise UserWarning("Unsupported conversion 'to {0!s}'".format(from_.name))
            return f(out)

        if from_ == eColorSpace.HSV:
            if to == eColorSpace.Grey:
                out.append(HSVtoGrey(I))
            elif to == eColorSpace.BGR:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_HSV2BGR))
            elif to == eColorSpace.RGB:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_HSV2RGB))    
            elif to == eColorSpace.HSV:
                out.append(I)
            else:
                raise UserWarning("Unsupported conversion 'to {0!s}'".format(from_.name))
            return f(out)

        if from_ == eColorSpace.RGB:
            if to == eColorSpace.Grey:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_RGB2GRAY))
            elif to == eColorSpace.BGR:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_RGB2BGR))
            elif to == eColorSpace.HSV:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_RGB2HSV))  
            elif to == eColorSpace.RGB:
                out.append(I)
            else:
                raise UserWarning("Unsupported conversion 'to {0!s}'".format(from_.name))
            return f(out)

        if from_ == eColorSpace.Grey:
            if to == eColorSpace.Grey:
                out.append(I)
            elif to == eColorSpace.BGR:
                out.append(_cv2.cvtColor(I, _cv2.COLOR_GRAY2BGR))
            elif to == eColorSpace.HSV:
                tmp = _cv2.cvtColor(I, _cv2.COLOR_GRAY2BGR)
                out.append(_cv2.cvtColor(tmp, _cv2.COLOR_BGR2HSV))    
            else:
                raise UserWarning("Unsupported conversion 'to {0!s}'".format(from_.name))
            return f(out)
    
        raise UserWarning("Unsupported conversion 'from {0!s}'".format(from_.name))



def split_channels(img):
    '''(str|ndarray) -> list:ndarray

    Given an image of n channels,
    splits channels into list elements, if
    img was OpenCV, this will be BGRA
    
    img:
        path to an image or an ndarray of arbitary depth

    Returns:
        List of ndarrays, with each element representing
        a channel
    '''
    return _np.dsplit(img, len(img.shape))



def make_cmap(name, n=256):
    '''make a cmap'''
    data = _CMAP_DATA[name]
    xs = _np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x + eps]
            yp += [y1, y2]
        ch = _np.interp(xs, xp, yp)
        channels.append(ch)
    return _np.uint8(_np.array(channels).T * 255)
