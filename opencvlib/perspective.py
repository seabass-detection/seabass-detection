# pylint: disable=C0103, too-few-public-methods, locally-disabled,
# no-self-use, unused-argument
'''edge detection and skeletonization
'''

#Link to integration
#https://stackoverflow.com/questions/13320262/calculating-the-area-under-a-curve-given-a-set-of-coordinates-without-knowing-t
#https://www.khanacademy.org/math/ap-calculus-ab/integration-applications-ab/average-value-of-a-function-ab/v/average-function-value-closed-interval
#simpsons rule

class Camera():
    '''just a container for camera properties
    '''
    def __init__(self, f, px_x, px_y, x_mm, y_mm):
        self.f = f
        self.px_x = px_x
        self.px_y = px_y
        self.x_mm = x_mm
        self.y_mm = y_mm


class Measure():
    '''a measure, just a variable container'''
    def __init__(self, lens_subj_dist=None, marker_length_mm=None, marker_length_px=None):
        self.lens_subj_dist = lens_subj_dist
        self.marker_length_mm = marker_length_mm
        self.marker_length_px = marker_length_px


def get_perspective_correction(bg_dist, object_depth, length):
    '''(float, float)->float|None
    Return the length corrected for the depth of the object
    considering the backplane of the object to be the best
    representative of the length
    *NOTE* The length of the object has been accurately measured
    '''
    if bg_dist is None or object_depth is None or length is None:
        return None
    elif bg_dist == 0 or 1 - (object_depth / bg_dist) == 0:
        return None

    return length / (1 - (object_depth / bg_dist))


def get_perspective_correction_iter_linear(coeff,
                                           const,
                                           bg_dist,
                                           length,
                                           profile_mean_height=1,
                                           last_length=0,
                                           stop_below_proportion=0.01):
    '''(float, float, float, float,float)->float|None
    Return the length corrected for the depth of the object
    considering the backplane of the object to be the best
    representative of the length.
    *NOTE* The length of the object was itself estimated from the foreground standard measure

    Coeff and constant are used to calculate an objects depth from its length
    The object depth is used to create a iterative series sum which add to the length
    to return the sum of lengths once the last length added was less then stop_below

    stop_below_proportion is the stopping criteria, once the last
    calculated length to add is is less than last_length*stop_below_proportion
    we return the result and stop the iteration
    '''
    if bg_dist == 0 or bg_dist is None or coeff == 0 or coeff is None or length is None:
        return None

    if last_length == 0:
        object_depth = length * coeff + const
    else:
        object_depth = last_length * coeff + const

    if object_depth <= 0:
        return length
    elif length == 0:
        return 0
    elif (last_length / length < stop_below_proportion) and last_length > 0:
        return length

    if last_length == 0:  # first call
        l = get_perspective_correction(bg_dist, object_depth, length) - length
    else:
        l = get_perspective_correction(bg_dist, object_depth, last_length) - last_length

    if l is None:
        return None

    return get_perspective_correction_iter_linear(coeff, const, bg_dist, length + (l * profile_mean_height), (l * profile_mean_height), stop_below_proportion)


def subjdist_knowndist(Known, Unknown):
    '''(Class:Measure, Class:Measure) -> float|None
    Get subject-lens distance
    estimate from a photograph of known distance
    with fiducial marker of known length
    '''
    #https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    assert isinstance(Known, Measure)
    assert isinstance(Unknown, Measure)
    x = [Known.marker_length_px, Known.lens_subj_dist, Known.marker_length_mm, Unknown.marker_length_mm, Unknown.marker_length_px]
    if not all(x):
        return None
    if Known.marker_length_mm == 0 or Unknown.marker_length_px == 0:
        return None
    F = Known.marker_length_px * Known.lens_subj_dist / Known.marker_length_mm
    return Unknown.marker_length_mm * F / Unknown.marker_length_px


def subjdist_camera(Cam, Unknown):
    '''(Class:Camera, Class:Measure) -> float|None

    Estimate lens-subject distance from the camera properties
    and the known marker length in mm and measure marker pixel
    length

    Currently assumes just using the width and not the height.

    Camera properties needed are the:
    Real cmos width in mm
    The cmos width in pixels
    The cameras focal length in mm
    '''
    assert isinstance(Cam, Camera)
    assert isinstance(Unknown, Measure)
    x = [Cam.f, Unknown.marker_length_mm, Cam.px_x, Unknown.marker_length_px, Cam.x_mm]
    if not all(x):
        return None

    return (Cam.f * Unknown.marker_length_mm * Cam.px_x) / (Unknown.marker_length_px * Cam.x_mm)
