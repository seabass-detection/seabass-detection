#pylint: skip-file
#Dont remove any functions from here
'''
From opencv demos
This module contains some common routines used by other opencv demos.
'''

# Python 2/3 compatibility
import sys as _sys
from functools import reduce as _reduce
from warnings import warn as _warn
import numpy as _np
import cv2 as _cv2

# built-in modules
import os as _os
import itertools as _it
from contextlib import contextmanager as _contextmanager
from opencvlib import getimg as _getimg
from funclib.baselib import tuple_add_elementwise as _tupadd
import funclib.iolib as _iolib
import opencvlib.info as _info
from opencvlib.geom import centroid #leave as is
from opencvlib.geom import order_points as _order_points

IMAGE_EXTENSIONS_DOTTED = ['.bmp', '.jpg', '.jpeg',
                    '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

IMAGE_EXTENSIONS_SANS_DOT = ['bmp', 'jpg', 'jpeg',
                    'png', 'tif', 'tiff', 'pbm', 'pgm', 'ppm']

class Bunch(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __str__(self):
        return str(self.__dict__)


def splitfn(fn):
    path, fn = _os.path.split(fn)
    name, ext = _os.path.splitext(fn)
    return path, name, ext


def anorm2(a):
    return (a * a).sum(-1)


def anorm(a):
    return _np.sqrt(anorm2(a))


def homotrans(H, x, y):
    xs = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    ys = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    s = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return xs / s, ys / s


def overlay(l_img, s_img, x_offset, y_offset, s_img_alpha_px=(255, 255, 255)):
    '''(ndarray, ndarray, int, int, 3-tuple) -> ndarray
    Give two 3 channel images, overlay s_img
    onto l_img.

    The alpha channel of s_imgis is s_img_alpha_px

    l_img, s_image:
        image
    x_offset, y_offset:
        where to position s_img on l_img, CVXY coords.
    s_img_alpha_px:
        alpha channel pixels, i.e these will be transparent

    Returns:
        l_img, overlayed with s_img
    '''
    y1, y2 = y_offset, y_offset + s_img.shape[0]
    x1, x2 = x_offset, x_offset + s_img.shape[1]
    assert isinstance(l_img, _np.ndarray)
    assert isinstance(s_img, _np.ndarray)

    ch1 = _np.ones(s_img.shape[0:2]) * s_img_alpha_px[0]
    ch2 = _np.ones(s_img.shape[0:2]) * s_img_alpha_px[1]
    ch3 = _np.ones(s_img.shape[0:1]) * s_img_alpha_px[2]
    m1 = s_img[..., 0] == s_img_alpha_px[0]
    m2 = s_img[..., 1] == s_img_alpha_px[1]
    m3 = s_img[..., 2] == s_img_alpha_px[2]
    mask =  _np.logical_and.reduce((m1, m2, m3)) #logical_and only accepts 2 arrays
    alpha = _np.array(mask, dtype='uint8') * 255 #0 should keep background, 1 should keep foreground
    foreground = _np.dstack(s_img, alpha)

    alpha_s = foreground[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        l_img[y1:y2, x1:x2, c] = (alpha_s * foreground[:, :, c] +
                                  alpha_l * l_img[y1:y2, x1:x2, c])
    return l_img



def to_rect(a):
    a = _np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return _np.array(a, _np.float64).reshape(2, 2)


def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = _np.float64([[cx,  0, tx], [0, cy, ty], [0,  0,  1]])
    return M


def lookat(eye, target, up=(0, 0, 1)):
    fwd = _np.asarray(target, _np.float64) - eye
    fwd /= anorm(fwd)
    right = _np.cross(fwd, up)
    right /= anorm(right)
    down = _np.cross(fwd, right)
    R = _np.float64([right, down, fwd])
    tvec = -_np.dot(R, eye)
    return R, tvec


def mtx2rvec(R):
    w, u, vt = _cv2.SVDecomp(R - _np.eye(3))
    p = vt[0] + u[:, 0] * w[0]    # same as _np.dot(R, vt[0])
    c = _np.dot(vt[0], p)
    s = _np.dot(vt[1], p)
    axis = _np.cross(vt[0], vt[1])
    return axis * _np.arctan2(s, c)


def draw_str(dst, x, y, s, color=(255, 255, 255), scale=1.0, thickness=1, fnt=_cv2.FONT_HERSHEY_COMPLEX_SMALL, bottom_left_origin=False, box_background=None, box_pad=5, centre_box_at_xy=False):
    '''(byref:ndarray, int, int, str, 3:tuple, float, int, bool, 3-tuple|int|None) -> void
    Draw text on dst

    dst:
        Image to draw text on, byref
    x, y:
        draw at this CVXY point
    s:
        the text to draw
    color:
        text color as 3-tuple, e.g. black=(0, 0, 0)
    scale:
        sizing scale
    bottom_left_origin:
        x,y is XY, not CVXY
    box_background:
        color of box background in which to draw text,
        e.g. (255, 255, 255), otherwise no
        background is used around text.
    '''
    if isinstance(box_background, int):
        box_background = (box_background, ) * 3

    if dst is None:
        return None
    x = int(x); y = int(y)
    thickness = int(thickness)
    pt, baseline = _cv2.getTextSize(s, fnt, scale, thickness)
    w, h = pt
    #pt is x, y
    if box_background:
        assert len(dst.shape) == 2 or len(dst.shape) == 3, 'Expected the image to be 2 or 3 dimensions, got %s.' % len(dst.shape)
        if len(dst.shape) == 2:
            box = _np.ones((pt[1] + box_pad*2, pt[0] + box_pad*2))*box_background[0]
        else:
            box = _np.ones((pt[1] + box_pad*2, pt[0] + box_pad*2, 3))*box_background

        textOrg = (int((box.shape[1] - w)/2), int((box.shape[0] + h)/2))

        corner = tuple(int(z) for z in _tupadd((textOrg, (0, baseline))))
        wh = tuple(int(z) for z in _tupadd((textOrg, (w, -h))))

        _cv2.rectangle(box, corner, wh, box_background)
        _cv2.putText(box, s, textOrg, fnt,
                    scale, color, thickness=thickness, lineType=_cv2.LINE_AA)

        if centre_box_at_xy:
            yadj = int(box.shape[0] / 2)
            xadj = int(box.shape[1] / 2)
        else:
            yadj = 0; xadj = 0

        yy = box.shape[0] + y - yadj
        xx = box.shape[1] + x - xadj
        if yy <= dst.shape[0] and xx <= dst.shape[1]:
            dst[y - yadj: yy, x - xadj: xx] = box
        else:
            _warn('Text box in draw_str too big for the image.')
    else:
        _cv2.putText(dst, s, (x, y), fnt,
                    scale, color, thickness=thickness, lineType=_cv2.LINE_AA, bottomLeftOrigin=bottom_left_origin)


class Sketcher:
    def __init__(self, windowname, dests, colors_func):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        _cv2.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        _cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, dummy):
        pt = (x, y)
        if event == _cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == _cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & _cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                _cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# palette data from matplotlib/_cm.py
_jet_data = {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89, 1, 1),
                       (1, 0.5, 0.5)),
             'green': ((0., 0, 0), (0.125, 0, 0), (0.375, 1, 1), (0.64, 1, 1),
                       (0.91, 0, 0), (1, 0, 0)),
             'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65, 0, 0),
                       (1, 0, 0))}

cmap_data = {'jet': _jet_data}


def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = _np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x + eps]
            yp += [y1, y2]
        ch = _np.interp(xs, xp, yp)
        channels.append(ch)
    return _np.uint8(_np.array(channels).T * 255)


def nothing(*dummy1, **dummy2):
    pass


def clock():
    '''
    returns absolute time in seconds since
    ticker started (usually when OS started
    '''
    return _cv2.getTickCount() / _cv2.getTickFrequency()


@_contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock() - start) * 1000))


class StatValue:
    def __init__(self, smooth_coef=0.5):
        self.value = None
        self.smooth_coef = smooth_coef

    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = self.smooth_coef * self.value + (1.0 - self.smooth_coef) * v


class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        _cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None

    def onmouse(self, event, x, y, flags):
        x, y = _np.int16([x, y])  # BUG
        if event == _cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & _cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = _np.minimum([xo, yo], [x, y])
                x1, y1 = _np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)

    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        _cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def dragging(self):
        return self.drag_rect is not None


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = _it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = _it.izip_longest(fillvalue=fillvalue, *args)
    return output


def chessboard(patch_sz=100, col_first_patch=(0, 0, 0), col_second_patch=(255, 255, 255), cols=9, rows=6):
    '''(int, 3-tuple, 3-tuple, int, int, bool) -> ndarray
    Returns a chessboard pattern.
    The nr. of vertices = (cols-1) * (rows-1)

    patch_sz:
        Size of patches in pixels
    col_first_patch:
        colour of starting patch (top left)
    col_second_light:
        colour of next patch
    cols:
        number of patch columns
    rows:
        number of patch rows
    '''
    patch_sz = int(patch_sz)
    cols = int(cols)
    rows = int(rows)
    color = col_first_patch
    board = _np.zeros([patch_sz*cols, patch_sz*rows, 3]).astype('uint8')
    for i in range(0, (rows+1)*patch_sz, patch_sz):
        for j in range(0, (cols+1)*patch_sz, patch_sz):
            board[j:j+patch_sz, i:i+patch_sz, :1] = color[0]
            board[j:j+patch_sz, i:i+patch_sz, :2] = color[1]
            board[j:j+patch_sz, i:i+patch_sz, :3] = color[2]
            color = col_second_patch if color==col_first_patch else col_first_patch
    board = _cv2.rotate(board, _cv2.ROTATE_90_CLOCKWISE) #hack coz got r and c crossed
    return board


def draw_scale(img, x_break=50, y_break=50, tick_colour=(0, 0, 0), tick_length=10, x_offset=0, y_offset=0):
    '''(ndarray, int, int, 3-tuple, int, int) -> ndarray
    Draw scale on on image

    x_break:
        x main increment (cols)
    y_break:
        y main increment (rows)
    tick_colour:
        tuple representing tick colour, e.g. (255, 0, 0) is blue
    x_offset:
        start drawing at x_offset
    y_offset:
        start drawing at y_offset
    Returns:
        the image with ticks
    '''
    x = x_offset
    y = y_offset
    label_pad = 4
    yy, xx = img.shape[:2]
    while y < yy + y_offset:
        pts = [[0, y], [tick_length, y]]
        img = draw_line(pts, img, tick_colour)
        lbl = '%s' % (y - y_offset)
        lbl_pt = (tick_length + label_pad, y)
        _cv2.putText(img, lbl, lbl_pt, _cv2.FONT_HERSHEY_PLAIN, 0.8, tick_colour)
        y += y_break

    while x < xx + x_offset:
        pts = [[x, yy], [x, yy - tick_length]]
        img = draw_line(pts, img, tick_colour)
        lbl = '%s' % (x - x_offset)
        lbl_pt = (x, yy - tick_length - label_pad)
        _cv2.putText(img, lbl, lbl_pt, _cv2.FONT_HERSHEY_PLAIN, 0.8, tick_colour)
        x += x_break

    return img


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=_cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or CV_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        _cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        _cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep


def draw_line(pts, img, line_color=(0, 0, 0), label=False, label_color=(0, 0, 0)):
    '''(ndarray, 3-tuple, bool, 3-tuple) -> ndarray
    Plots a line, defined by a two points.

    pts:
        array of points in format CVXY, e.g. [[1,2], [3,4], ...]
    line_color:
        Colour of line
    label:
        Label the points with their coordinates
    label_color:
        Colour of label text, if pt_labels is True

    Returns:
        image (ndarray)
    '''
    poly_pts = _np.array(pts, dtype='int32')
    poly_pts = poly_pts.reshape((-1, 1, 2))
    _cv2.polylines(img, [poly_pts], True, line_color)

    if label:
        for i, pt in enumerate(pts):
            lbl = '%s, %s' % (int(pts[i][0]), int(pts[i][1]))
            _cv2.putText(img, lbl, (int(pt[0]) + x_offset, int(pt[1]) + y_offset), _cv2.FONT_HERSHEY_PLAIN, 0.8, label_color)
    return img


def draw_polygon(img, points, color=(0, 255,0), thickness=1):
    '''(ndarray, tuple|list) -> ndarray
    Draw a polygon
    '''
    #[10,5],[20,30],[70,20],[50,10]
    assert isinstance(img, _np.ndarray)
    cp = img.copy()
    pts_cp = _order_points(points)
    pts_cp = _np.array(points).astype('int32')
    pts_cp = pts_cp.reshape((-1, 1, 2))
    return _cv2.polylines(cp, [pts_cp], isClosed=True, color=color, thickness=thickness)


def draw_points(pts, img=None, x=None, y=None, join=False, line_color=(0, 0, 0), thickness=1, show_labels=True, label_color=(0, 0, 0), padding=0, add_scale=True,  radius=6, plot_centroid=False):
    '''(array, ndarray|str, int, int, bool, 3-tuple, bool, int) -> ndarray
    Show roi points largely for debugging purposes

    pts:
        array of points in format [[1,2], [3,4], ...]
    img:
        ndarray, path or none. If none then the points
        will be drawn on a white canvas of size determined
        by the points
    x:
        image width if no img is provided
    y:
        image height if no img is provided
    join:
        join the points with lines
    line_color:
        color of lines used if join=True
    thickness:
        thickness of line of joining dots
    show_labels:
        label points with their coordinates
    pad:
        padding to add around the image if img was None,
        or if image is not none, then this is the assumed
        padding used when img was first created.

    Returns:
        The image (or blank canvas) with points plotted
    '''
    if isinstance(pts, _np.ndarray):
        pts = pts.squeeze()
        pts = pts.squeeze()
        pts = pts.squeeze()
        pts = pts.tolist()

    #get size of display frame
    pad = padding
    new_image = img is None
    if img is None:
        xs, ys = zip(*pts)
        x = max(xs) + pad*2 if x is None or max(xs) < x else x
        y = max(ys) + pad*2 if y is None or max(ys) < y else y
        x = int(x)
        y = int(y)
        img_process = _np.ones((y, x, 3), dtype='uint8')*255
    else:
        img = _getimg(img)
        img_process = _np.copy(img)

    if plot_centroid:
        pt_mean = centroid(pts, _np.int)

    #added padding, so have to adjust points slightly
    if pad > 0:
        pts_padded = [[pt[0] + pad, pt[1] + pad] for pt in pts]
        if plot_centroid:
            pt_mean = [pt_mean[0] + pad, pt_mean[1] + pad]
        #draw original boundaries first time
        if new_image:
            _cv2.rectangle(img_process, (pad, pad), (img_process.shape[1] - pad, img_process.shape[0] - pad), (0, 0, 0))
    else:
        pts_padded = pts


    for i, pt in enumerate(pts_padded):
        centre = (int(pt[0]), int(pt[1]))
        _cv2.circle(img_process, centre, radius, (255, 255, 255), -11)
        _cv2.circle(img_process, centre, int(radius*1.1), (0, 0, 255), 1)
        _cv2.ellipse(img_process, centre, (radius, radius), 0, 0, 90, (0, 0, 255), -1)
        _cv2.ellipse(img_process, centre, (radius, radius), 0, 180, 270, (0, 0, 255), -1)
        _cv2.circle(img_process, centre, 1, (0, 255, 0), 1)


        if show_labels:
            lbl = '%s, %s' % (int(pts[i][0]), int(pts[i][1])) #use original point as label if we padded
            _cv2.putText(img_process, lbl, (int(pt[0]) + 10, int(pt[1]) - 10), _cv2.FONT_HERSHEY_PLAIN, 0.8, label_color)

    if plot_centroid:
        xx = (pt_mean[0] - radius, pt_mean[0] + radius)
        yy = (pt_mean[1] - radius, pt_mean[1] + radius)
        _cv2.line(img_process, (pt_mean[0], yy[0]), (pt_mean[0], yy[1]), (0, 255, 0))
        _cv2.line(img_process, (xx[0], pt_mean[1]), (xx[1], pt_mean[1]), (0, 255, 0))

    if join:
        pts_padded = _order_points(pts_padded)
        poly_pts = _np.array(pts_padded, dtype='int32')
        poly_pts = poly_pts.reshape((-1, 1, 2))
        _cv2.polylines(img_process, [poly_pts], True, line_color, thickness=thickness)

    if add_scale:
        img_process = draw_scale(img_process, y_offset=padding, x_offset=padding)

    return img_process
