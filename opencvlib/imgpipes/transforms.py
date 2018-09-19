# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import
'''transforms compatible with generators Transform framework'''


from opencvlib.transforms import gamma, gamma1, log, sigmoid
from opencvlib.transforms import compute_average2, equalize_adapthist, equalize_hist
from opencvlib.transforms import histeq, histeq_adapt, histeq_color
from opencvlib.transforms import intensity, resize
from opencvlib.transforms import Transforms, Transform

from opencvlib.color import BGR2RGB, RGB2BGR, togreyscale
