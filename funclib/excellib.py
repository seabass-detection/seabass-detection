'''excel functions with xlwings'''
from warnings import warn as _warn

from funclib import get_platform
import numpy

if get_platform() != 'windows':
    try:
        import xlwings
    except Exception as e:
        _warn('xlwings not installed. This module will not function')


def numpy_pickle_view(picklepath):
    '''(str) -> void
    Loads and shows a pickled numpy array on the file system
    in excel
    '''
    try:
        arr = numpy.load(picklepath)
        xlwings.view(arr)
    except Exception as _:
        pass
