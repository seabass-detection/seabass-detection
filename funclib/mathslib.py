'''pure maths helper functions'''

from scipy.integrate import quad


def function_mean_over_interval(func, lower, upper):
    '''(function, float, float)->float
    Given a function return the function mean between the lower and upper interval
    Lower or upper will accept numpy.inf
    '''
    return quad(func, lower, upper)[0] / (upper - lower)
