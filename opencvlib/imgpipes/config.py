'''
Set the config variable.
'''

import configparser as _cp
from inspect import getsourcefile as _getsourcefile
import os.path as _path
from warnings import warn as _warn
#import json

from funclib.iolib import fixp as _fixp
from funclib.iolib import get_file_parts2 as _get_file_parts2


def _cfg_path():
    '''()->str
    returns full path to config file
    '''
    pth = _get_file_parts2(_path.abspath(_getsourcefile(lambda: 0)))[0]
    pth = _fixp(pth)
    return _fixp(_path.join(pth, 'imgpipes.cfg'))



_config = _cp.RawConfigParser()
_config.read(_cfg_path())

try:
    digikamdb = _fixp(_config.get("DIGIKAM", "dbpath"))
except Exception as _:
    _warn('Could not read digikamdb path, the command was _config.get("DIGIKAM", "dbpath")')

try:
    VOC_ROOT_DIR = _fixp(_config.get("VOC_UTILS", "root_dir"))
    VOC_IMG_DIR = _fixp(_config.get("VOC_UTILS", "img_dir"))
    VOC_ANN_DIR = _fixp(_config.get("VOC_UTILS", "ann_dir"))
    VOC_SET_DIR = _fixp(_config.get("VOC_UTILS", "set_dir"))
    if VOC_SET_DIR == '' or VOC_ANN_DIR == '' or VOC_IMG_DIR == '' or VOC_ROOT_DIR == '':
        _warn('Could not read paths for voc_utils from config file %s' % _cfg_path())
except Exception as _:
    _warn('Could not read paths for voc_utils.py from config file %s' % _cfg_path())



#orientations = config.getint("hog", "orientations")
#cells_per_block = json.loads(config.get("hog", "cells_per_block"))
#normalize = config.getboolean("hog", "normalize")
#threshold = config.getfloat("nms", "threshold")
