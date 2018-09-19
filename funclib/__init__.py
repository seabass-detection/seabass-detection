'''main init for funclib'''


__all__ = [
    'arraylib',
    'baselib',
    'configlib',
    'excellib',
    'inifilelib',
    'iolib',
    'jsonlib',
    'log',
    'numericslib',
    'pandaslib',
    'shell_lib',
    'statslib',
    'stopwatch',
    'stringslib']


from funclib.baselib import get_platform
from funclib.baselib import isPython2
from funclib.baselib import isPython3
from funclib.baselib import isIterable
