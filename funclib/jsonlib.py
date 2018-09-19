# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument
'''json helper functions'''

import os.path as _path
import json as _json
#import jmespath as _jmes

#from funclib.baselib import odict

def load(filepath, dicttype=dict):
    '''(str, object)->dict
    Load the VGG JSON file and return the dict as type
    specified by dicttype

    Custom dictionary types allow overriding of
    default behaviour, such as partial key matching
    for retreival of dictionary values.

    Custom dictionary classes are in baselib
    '''
    pth = _path.normpath(filepath)
    with open(pth) as data_file:
        return dicttype(_json.load(data_file))
