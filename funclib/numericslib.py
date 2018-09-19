'''basic number related helper functions'''
import math as _math


def translate_scale(val_in, out_min, out_max, val_in_max):
    '''(float, float, float, float) -> float
    Translate val_in to a different scale range.

    val_in: the value to convert to new range
    out_min: the minimum value of the target range
    out_max: the max value of the target range
    val_in_max: the maximum value of the input range

    Example:
    Standardise a welsh city population value to lie between 0 and 1
    Bangor population = 5000, maximum population=100,000
    >>>translate_scale(5000, 0, 1, 100000)
    0.05
    '''
    return val_in*(out_max - out_min)*(1/val_in_max) + out_min


def is_float(test):
    '''(any) -> bool
    Return true of false if s is a float
    '''
    try:
        float(test)
        return True
    except ValueError:
        return False


def roundx(v):
    '''(float)->float
    round to the more extreme value
    '''
    if v < 0:
        return _math.floor(v)
    return _math.ceil(v)
