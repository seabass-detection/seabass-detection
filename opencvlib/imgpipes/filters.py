# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
'''filters to be consumed in generators.

All functions should return a boolean and receive an ndarray (image) as the first argument.

A return boolean of True, would keep the image, false indicated it should not be
yielded from the generator

Filters
'''
import funclib.baselib as _baselib

import opencvlib.decs as _decs
import opencvlib.info as _info
from opencvlib import getimg as _getimg




#region Handling filters for generators, ie rules for 'dropping' an image
#from the processing chain
class Filter():
    '''this is a filter'''
    def __init__(self, func, *args, **kwargs):
        '''(method|function, arguments)->bool
        the function and the arguments to be applied
        to the function to evaluate that function
        for an image (ndarray).

        All functions must return a boolean.
        Compatible filter functions are imported into imgpipes.filters
        '''
        self._args = args
        self._kwargs = kwargs
        self._func = func
        self.valid = True


    @_decs.decgetimgmethod
    def imgisvalid(self, img):

        '''(str|ndarray)->bool
        Perform the transform on passed image.
        Returns the transformed image and sets
        to class instance variable img_transformed
        '''
        self.valid = self._func(img, *self._args, **self._kwargs)
        return self.valid


class Filters():
    '''this handles building search filters
    to apply to an image
    '''
    def __init__(self, *args, img=None):
        '''(str|ndarray, Filter functions)
        '''
        self.img = _getimg(img)
        self.fQueue = []
        self.fQueue.extend(args)


    def add(self, *args):
        '''(Filter(s))->void
        Queue one or more filters.
        Order is FIFO
        '''
        self.fQueue.extend(args)


    @_decs.decgetimgmethod
    def validate(self, img=None):
        '''(str|ndarray)->ndarray
        perform the transformations. Is FIFO.
        Set img_transformed property, and returns
        the transformed image
        '''
        if not img is None:
            self.img = img

        if _baselib.isempty(self.fQueue):
            return True

        for F in self.fQueue:
            assert isinstance(F, Filter)
            if not F.imgisvalid(self.img):
                return False
        return True
#endregion

is_higher_res = _info.ImageInfo.is_higher_res
is_lower_res = _info.ImageInfo.is_lower_res
isbw = _info.ImageInfo.isbw
