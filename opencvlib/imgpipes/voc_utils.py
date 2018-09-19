# pylint: disable=C0302, line-too-long, too-few-public-methods, too-many-branches, too-many-statements, unused-import, no-member
'''Interact with PASCAL VOC 2012 images and metadata
'''
from inspect import getsourcefile as _getsourcefile
from os.path import abspath as _abspath
from os.path import join as _join
from os.path import normpath as _normpath
import os as _os


import pandas as _pd
from bs4 import BeautifulSoup as _BS
from more_itertools import unique_everseen  as _unique_everseen
import numpy as _np
import matplotlib.pyplot as _plt
import skimage as _skimage
from skimage import io as _io

from funclib.stringslib import trim as _trim
import funclib.inifilelib as _inifilelib
import funclib.iolib as _iolib
import opencvlib.imgpipes.config as _cfg
import opencvlib.roi as _roi


_SET_DIR = _cfg.VOC_SET_DIR
_ANN_DIR = _cfg.VOC_ANN_DIR
_ROOT_DIR = _cfg.VOC_ROOT_DIR
_IMG_DIR = _cfg.VOC_IMG_DIR


def get_fname(category, datatype):
    '''get full file name'''
    s = category + '_' + datatype + '.txt'
    return _os.path.normpath(_os.path.join(_cfg.VOC_SET_DIR, s))


def list_image_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']


def imgs_from_category(cat_name, dataset):
    """
    Summary

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        dataset (string): "train", "val", "train_val", or "test" (if available)

    Returns:
        pandas dataframe: pandas DataFrame of all filenames from that category
    """
    filename = get_fname(cat_name, dataset)
    df = _pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename', 'true'])
    return df


def imgs_from_category_as_list(cat_name, dataset):
    """
    Get a list of filenames for images in a particular category
    as a list rather than a pandas dataframe.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        dataset (string): "train", "val", "train_val", or "test" (if available)

    Returns:
        list of srings: all filenames from that category
    """
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values


def regions_generator(fullfilename, category=''):
    '''(str, str) -> str|None, list|None
    Yields bounding rectangles of regions for the defined image.

    fullfilename:
        the full path to the file, e.g. c:/temp/img.jpg
    category:
        If not empty, will only yield bounding boxes of objects of defined category

    Returns:
        object category and the bounding box points in CVXY format
    '''
    _, file_key, _ = _iolib.get_file_parts2(fullfilename)
    anno = load_annotation(file_key)
    objs = anno.findAll('object')
    for obj in objs:
        obj_names = obj.findChildren('name')
        for name_tag in obj_names:
            if category == '' or (category != '' and str(name_tag.contents[0]) == category):
                _ = anno.findChild('filename').contents[0]
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])
                pts = _roi.points_convert([xmin, xmax, ymin, ymax],
                    0, 0, _roi.ePointConversion.XYMinMaxtoCVXY, _roi.ePointsFormat.XY)
                yield str(name_tag.contents[0]), pts



def annotation_file_from_img(img_name):
    """
    Given an image name, get the annotation file for that image

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        string: file path to the annotation file
    """
    s = _trim(img_name, '.jpg')
    s = _trim(s, '.JPG')
    return _os.path.normpath((_os.path.join(_ANN_DIR, s) + '.xml'))


def load_annotation(img_filename):
    """
    Load annotation file for a given image.

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        _bs structure: the annotation labels loaded as a
            _bs data structure
    """
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return _BS(xml, 'lxml')


def load_img(img_filename):
    """
    Load image from the filename. Default is to load in color if
    possible.

    Args:
        img_name (string): string of the image name, relative to
            the image directory.

    Returns:
        np array of float32: an image as a numpy array of float32
    """
    img_filename = _os.path.normpath(_os.path.join(_IMG_DIR, img_filename + '.jpg'))
    img = _skimage.img_as_float(_io.imread(
        img_filename)).astype(_np.float32)
    if img.ndim == 2:
        img = img[:, :, _np.newaxis]
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def load_imgs(img_filenames):
    """
    Load a bunch of images from disk as np array.

    Args:
        img_filenames (list of strings): string of the image name, relative to
            the image directory.

    Returns:
        np array of float32: a numpy array of images. each image is
            a numpy array of float32
    """
    return _np.array([load_img(fname) for fname in img_filenames])


def _load_data(category, data_type=None):
    """
    Loads all the data as a pandas DataFrame for a particular category.

    Args:
        category (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Raises:
        ValueError: when you don't give "train" or "val" as data_type

    Returns:
        pandas DataFrame: df of filenames and bounding boxes
    """
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    to_find = category
    filename = get_fname(category, data_type)
    filename_csv = (filename + '.csv')

    if _os.path.isfile(filename_csv):
        return _pd.read_csv(filename_csv)

    train_img_list = imgs_from_category_as_list(to_find, data_type)
    data = []
    for item in train_img_list:
        anno = load_annotation(item)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                if str(name_tag.contents[0]) == category:
                    fname = anno.findChild('filename').contents[0]
                    bbox = obj.findChildren('bndbox')[0]
                    xmin = int(bbox.findChildren('xmin')[0].contents[0])
                    ymin = int(bbox.findChildren('ymin')[0].contents[0])
                    xmax = int(bbox.findChildren('xmax')[0].contents[0])
                    ymax = int(bbox.findChildren('ymax')[0].contents[0])
                    data.append([fname, xmin, ymin, xmax, ymax])
    df = _pd.DataFrame(
        data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax'])
    df.to_csv(filename_csv)
    return df


def get_image_url_list(category, data_type=None):
    """
    For a given data type, returns a list of filenames.

    Args:
        category (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Returns:
        list of strings: list of all filenames for that particular category
    """
    makef = lambda fname: _os.path.normpath(_IMG_DIR + '/' + fname)
    df = _load_data(category, data_type=data_type)
    image_url_list = [makef(x) for x in list(_unique_everseen(list((df['fname']))))]
    return image_url_list


def get_masks(cat_name, data_type, mask_type=None):
    """
    Return a list of masks for a given category and data_type.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"
        mask_type (string, optional): either "bbox1" or "bbox2" - whether to
            sum or add the masks for multiple objects

    Raises:
        ValueError: if mask_type is not valid

    Returns:
        list of np arrays: list of np arrays that are masks for the images
            in the particular category.
    """
    # change this to searching through the df
    # for the bboxes instead of relying on the order
    # so far, should be OK since I'm always loading
    # the df from disk anyway
    # mask_type should be bbox1 or bbox
    if mask_type is None:
        raise ValueError('Must provide mask_type')
    df = _load_data(cat_name, data_type=data_type)
    # load each image, turn into a binary mask
    masks = []
    prev_url = ""
    blank_img = None
    for _, entry in df.iterrows():
        img_url = _os.path.normpath(_os.path.join(_IMG_DIR, entry['fname']))
        if img_url != prev_url:
            if blank_img is not None:
                # TODO: options for how to process the masks
                # make sure the mask is from 0 to 1
                max_val = blank_img.max()
                if max_val > 0:
                    min_val = blank_img.min()
                    # print "min val before normalizing: ", min_val
                    # start at zero
                    blank_img -= min_val
                    # print "max val before normalizing: ", max_val
                    # max val at 1
                    blank_img /= max_val
                masks.append(blank_img)
            prev_url = img_url
            img = load_img(img_url)
            blank_img = _np.zeros((img.shape[0], img.shape[1], 1))
        bbox = [entry['xmin'], entry['ymin'], entry['xmax'], entry['ymax']]
        if mask_type == 'bbox1':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1.0
        elif mask_type == 'bbox2':
            blank_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 1.0
        else:
            raise ValueError('Not a valid mask type')
    # TODO: options for how to process the masks
    # make sure the mask is from 0 to 1
    max_val = blank_img.max()
    if max_val > 0:
        min_val = blank_img.min()
        # print "min val before normalizing: ", min_val
        # start at zero
        blank_img -= min_val
        # print "max val before normalizing: ", max_val
        # max val at 1
        blank_img /= max_val
    masks.append(blank_img)
    return _np.array(masks)


def get_imgs(cat_name, data_type=None):
    """
    Load and return all the images for a particular category.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())
        data_type (string, optional): "train" or "val"

    Returns:
        np array of images: np array of loaded images for the category
            and data_type.
    """
    image_url_list = get_image_url_list(cat_name, data_type=data_type)
    imgs = []
    for url in image_url_list:
        imgs.append(load_img(url))
    return _np.array(imgs)


def display_image_and_mask(img, mask):
    """
    Display an image and it's mask side by side.

    Args:
        img (np array): the loaded image as a np array
        mask (np array): the loaded mask as a np array
    """
    _plt.figure(1)
    _plt.clf()
    ax1 = _plt.subplot(1, 2, 1)
    ax2 = _plt.subplot(1, 2, 2)
    ax1.imshow(img)
    ax1.set_title('Original image')
    ax2.imshow(mask)
    ax2.set_title('Mask')
    _plt.show(block=False)


def cat_name_to_cat_id(cat_name):
    """
    Transform a category name to an id number alphabetically.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())

    Returns:
        int: the integer that corresponds to the category name
    """
    cat_list = list_image_sets()
    cat_id_dict = dict(zip(cat_list, range(len(cat_list))))
    return cat_id_dict[cat_name]


def display_img_and_masks(
        img, true_mask, predicted_mask, block=False):
    """
    Display an image and it's two masks side by side.

    Args:
        img (np array): image as a np array
        true_mask (np array): true mask as a np array
        predicted_mask (np array): predicted_mask as a np array
        block (bool, optional): whether to display in a blocking manner or not.
            Default to False (non-blocking)
    """
    m_predicted_color = predicted_mask.reshape(
        predicted_mask.shape[0], predicted_mask.shape[1])
    m_true_color = true_mask.reshape(
        true_mask.shape[0], true_mask.shape[1])
    # m_predicted_color = predicted_mask
    # m_true_color = true_mask
    # plt.close(1)
    _plt.figure(1)
    _plt.clf()
    _plt.axis('off')
    _, (ax1, ax2, ax3) = _plt.subplots(nrows=1, ncols=3, num=1)
    ax1.get_xaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax3.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    ax3.get_yaxis().set_ticks([])

    ax1.imshow(img)
    ax2.imshow(m_true_color)
    ax3.imshow(m_predicted_color)
    _plt.draw()
    _plt.show(block=block)


def load_data_multilabel(data_type=None):
    """
    Returns a data frame for all images in a given set in multilabel format.

    Args:
        data_type (string, optional): "train" or "val"

    Returns:
        pandas DataFrame: filenames in multilabel format
    """
    if data_type is None:
        raise ValueError('Must provide data_type = train or val')
    filename = _os.path.normpath(_os.path.join(_SET_DIR, data_type + ".txt"))
    cat_list = list_image_sets()
    df = _pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename'])
    # add all the blank rows for the multilabel case
    for cat_name in cat_list:
        df[cat_name] = 0
    for info in df.itertuples():
        index = info[0]
        fname = info[1]
        anno = load_annotation(fname)
        objs = anno.findAll('object')
        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                tag_name = str(name_tag.contents[0])
                if tag_name in cat_list:
                    df.at[index, tag_name] = 1
    return df
