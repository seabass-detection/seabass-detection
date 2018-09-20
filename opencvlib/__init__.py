import os.path as _path
import cv2 as _cv2

IMAGE_EXTENSIONS = ('.bmp',
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.tif',
                    '.tiff',
                    '.pbm',
                    '.pgm',
                    '.ppm')

IMAGE_EXTENSIONS_AS_WILDCARDS = ('*.bmp',
                                 '*.jpg',
                                 '*.jpeg',
                                 '*.png',
                                 '*.tif',
                                 '*.tiff',
                                 '*.pbm',
                                 '*.pgm',
                                 '*.ppm')

__all__ = [	
			'distance',
           'geom',
           'keypoints'
           'perspective'
           'roi',
           'transforms']



def getimg(img, outflag=_cv2.IMREAD_UNCHANGED):
    '''(ndarray|str)->ndarray
    tries to load the image if its a path and returns the loaded ndarray
    otherwise returns input img if it is an ndarray

    Also consider using @decs._decs.decgetimg decorator
    '''
    if isinstance(img, str):
        return _cv2.imread(_path.normpath(img), outflag)

    return img
