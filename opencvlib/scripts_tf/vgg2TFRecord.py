# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use, unused-argument, unused-import
'''
Create a TFRecord file from images and their assigned roi. This saves the whole image into the TFRecord file.

    -d: Delete all .record files in the output folder first
    -b: batch size, create multiple .record files with a size specified with the -b argument
    -f: Will also add horizontally flipped images and flipped points
    Positional args:
        source_folder, output_folder, vgg_file_name


Example:
    vgg2TFRecord.py -d -b 20 "C:/candidate" "C:/candidate/test.record" vgg_body.json

Comments:
    output_folder will be created if it doesn't exist
'''

import argparse
import os.path as path
import os
from math import ceil
import io

from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

from object_detection.utils import dataset_util
import funclib.iolib as iolib
import funclib.stringslib as stringslib
import opencvlib.roi as roi
import opencvlib.imgpipes.vgg as vgg
from opencvlib.imgpipes.generators import VGGROI
import opencvlib.transforms as transforms
from opencvlib.view import show
from opencvlib import common

PP = iolib.PrintProgress()
vgg.SILENT = True

BAD_LIST = []

def image_is_invalid(imgpath):
    '''(str) -> bool
    Load an image and run some validation tests.
    Append errors to global BAD_LIST.

    Returns: True if image is invalid, else False
    '''
    global BAD_LIST

    img = Image.open(imgpath)
    invalid = False
    errs = ['%s: ' % imgpath]
    if img.format != 'JPEG':
        invalid = True
        errs.append('Format was %s. Expected jpeg' % img.format)

    np_im = np.array(img)
    if len(np_im.shape) != 3:
        invalid = True
        errs.append('Image had %s channels. Expected 3.' % len(np_im.shape))
    else:
        if np_im.shape[2] != 3:
            invalid = True
            errs.append('Image had %s channels. Expected 3.' % len(np_im.shape[2]))

    if invalid:
        errs.append('\n')
        BAD_LIST.append(' | '.join(errs))

    return invalid


def create_tf_example(filename, xmin, xmax, ymin, ymax):
    '''create'''
    filename = path.normpath(filename)

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    _, fname, _ = iolib.get_file_parts2(filename)
    filename = fname.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    xmins.append(xmin / width)
    xmaxs.append(xmax / width)
    ymins.append(ymin / height)
    ymaxs.append(ymax / height)
    classes_text.append('bass'.encode('utf8'))
    classes.append(1)#1 is the code for bass

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



def main():
    '''entry'''
    cmdline = argparse.ArgumentParser(description=__doc__)
    cmdline.add_argument('-d', '--delete', action='store_true', help='Delete tfrecord .record file(s) from the output folder.')
    cmdline.add_argument('-x', '--addx', default='1.0', type=float, help='Grow roi width by this proportion.')
    cmdline.add_argument('-y', '--addy', default='1.0', type=float, help='Grow roi height by this proportion.')
    cmdline.add_argument('-f', '--flip', action='store_true', help='Also add flipped images.')
    cmdline.add_argument('-b', '--batch_sz', help='Batch size.', default=1)
    cmdline.add_argument('source_folder', help='The folder containing the images and vgg file')
    cmdline.add_argument('output_file', help='The TFRecord file to create')
    cmdline.add_argument('vgg_file_name', help='The filename of the vgg file, must be in source folder')

    args = cmdline.parse_args()
    batch_sz = int(args.batch_sz)
    assert batch_sz > 0, 'Batch size cannot be 0 or less'

    src = path.normpath(args.source_folder)
    out = path.normpath(args.output_file)
    fld, _, _ = iolib.get_file_parts(out)
    if args.delete:
        for f in iolib.file_list_generator1(fld, '*.record'):
            iolib.files_delete2(f)

    vgg_file = path.normpath(src + '/' + args.vgg_file_name)

    fld, _, _ = iolib.get_file_parts2(out)
    iolib.create_folder(fld)
    errout = path.normpath(path.join(fld, 'bad_image_files.csv'))

    if iolib.file_exists(out):
        print('Output TFRecord %s already exists. Delete it manually.' % out)
        return

    Gen = VGGROI(vgg_file)
    filecnt = sum([1 for x in Gen.generate(path_only=True)])
    PP.max = filecnt

    if batch_sz <= 1:
        nr_parts = 1
        writer = tf.python_io.TFRecordWriter(out)
    else:
        nr_parts = ceil(filecnt / batch_sz)
    first = True
    has_errs = False
    for _, imgpath, dic in Gen.generate(grow_roi_x=args.addx, grow_roi_y=args.addy, path_only=False): #path_only has to be false so grow_roi gets capped
        if image_is_invalid(imgpath):
            PP.increment()
            continue
        img = cv2.imread(imgpath)
        ptsx, ptsy = list(zip(*dic['pts_grown_cvxy']))

        pts_flipped = roi.flip_points(dic['pts_grown_cvxy'], img.shape[0], img.shape[1], hflip=True)
        ptsx_flipped, ptsy_flipped = list(zip(*pts_flipped))

        #DEBUG STUFF
        #x = dic['region_attributes'].x
        #y = dic['region_attributes'].y
        #w = dic['region_attributes'].w
        #h = dic['region_attributes'].h
        #pts_orig = roi.points_convert([x, x + w, y, y + h], img.shape[1], img.shape[0], roi.ePointConversion.XYMinMaxtoCVXY, roi.ePointsFormat.XY)
        #img = common.draw_points(pts_orig, img, join=True, line_color=(0, 0, 0), thickness=2)
        #img = common.draw_points(dic['pts_grown_cvxy'], img, join=True, line_color=(255, 255, 255), thickness=2)
        #title = 'grow_x %.2f grow_y %.2f' % (args.addx, args.addy)
        #show(img, title=title)

        if nr_parts > 1:
            suffix = str(ceil(PP.iteration / batch_sz)).zfill(len(str(nr_parts)))
            fld, fname, ext = iolib.get_file_parts(out)
            fname = '%s-%s%s' % (fname, suffix, ext)
            batch_name = path.normpath(path.join(fld, fname))

            if PP.iteration % batch_sz == 1 and PP.iteration > 1:
                writer.close()
                writer = tf.python_io.TFRecordWriter(batch_name)
            elif first:
                writer = tf.python_io.TFRecordWriter(batch_name)
                first = False
        PP.increment()

        tf_example = create_tf_example(imgpath, min(ptsx), max(ptsx), min(ptsy), max(ptsy))
        writer.write(tf_example.SerializeToString())
        #hackey, flip image, save to temp folder, stick it in the tfrecord, then delete the image
        if args.flip:
            tmp = iolib.get_temp_fname(suffix='.jpg')
            imgflip = cv2.flip(img, flipCode=1) #x axis flip
            cv2.imwrite(tmp, imgflip)
            tf_example = create_tf_example(tmp, min(ptsx_flipped), max(ptsx_flipped), min(ptsy_flipped), max(ptsy_flipped))
            writer.write(tf_example.SerializeToString())
            try:
                iolib.files_delete2(tmp)
            except Exception as dummy:
                pass


    try:
        writer.close()
    except Exception as _:
        pass


    if has_errs:
        iolib.writecsv(errout, BAD_LIST, inner_as_rows=False)

if __name__ == "__main__":
    main()
