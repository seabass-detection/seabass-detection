# pylint: disable=C0302, line-too-long, too-few-public-methods

# Albumroots.specificpath is the root path of the album from the drive, excluding the drive
# e.g. /development/python/opencvlib/calibration for the calibration album
# Albums.relative path is the path after the album root
# e.g.

'''
various routines to work with the digikam library, stored in sqlite
'''
import os as _os
import warnings as _warnings

import cv2 as _cv2
import sqlite3 as _sqlite3

from funclib.baselib import get_platform as _get_platform
from funclib.iolib import get_available_drive_uuids as _get_available_drive_uuids
from funclib.stringslib import rreplace as _rreplace
from funclib.stringslib import add_left as _add_left
from funclib.iolib import print_progress as _print_progress

import dblib.sqlitelib as _sqlitelib

SPECIFIC_PATH_OVERRIDE = ''
SILENT = False


def read_specific_path(pathin):
    '''(str)->str
    Use SPECIFIC_PATH_OVERRIDE if defined, primarily
    for if we run on a different computer

    Replace specific path with specific_path_override
    '''
    if SPECIFIC_PATH_OVERRIDE == '':
        return pathin

    return SPECIFIC_PATH_OVERRIDE


class _ImagesValidator(object):
    '''Use this to process a list of digikam image paths.
    It processes the paths to generate valid and invalid links

    Pass a list of files during instance creation,
    or setting the image_files property to a list of files will also force a validation

    If cv_load_test is true, then validation will try and load images with _cv2.imread.
    '''

    def __init__(self, file_ptrs=None):
        '''files is a list of paths'''
        self._image_files = file_ptrs
        self._invalid_image_files = []
        self._valid_image_files = []
        self.cv_load_test = False

        if file_ptrs is not None:
            self._validate()

    def _validate(self):
        '''go through self._image_paths and assign valid and invalid paths to module variables'''
        self._invalid_image_files = []
        for images in self._image_files:
            if not _os.path.isfile(images):
                self._invalid_image_files.append(images)
            else:
                if self.cv_load_test:
                    try:
                        nd = _cv2.imread(images)
                        if nd is None:
                            nd = None
                            self._invalid_image_files.append(images)
                    except Exception:
                        self._invalid_image_files.append(images)

        self._valid_image_files = list(
            set(self._image_files).difference(self._invalid_image_files))

    @property
    def invalid_count(self):
        '''number of invalid images'''
        return len(self._invalid_image_files)

    @property
    def valid_count(self):
        '''number of invalid images'''
        return len(self._image_files) - len(self._invalid_image_files)

    @property
    def total_count(self):
        '''total number of images'''
        return len(self._image_files)

    @property
    def invalid(self):
        '''return a list of images which are invalid
        from the image_names list'''
        return self._invalid_image_files

    @property
    def valid(self):
        '''->list
        return list of valid images, these have just
        been validated as existing and no attempt
        has been made to load in opencv UNLESS image_generator
        was used to generate the valid and invalid lists
        '''
        return self._valid_image_files

    @property
    def image_files(self):
        '''image_names getter
        returns list of image names with full path'''
        return self._image_files

    @image_files.setter
    def image_files(self, image_files):
        '''(list, bool)->void
        resets list of files and does basic validation

        If self.cv_load_test is false then does not attempt load in OpenCV,
        if self.cv_load_test is true it will test to see if file can
        be loaded in opencv, this can take a long time!

        You can optionally use image_generator which as well
        as generating nd arrays via _cv2.imread()
        also set valid and invalid lists based on the
        success or imread()
        '''
        self._image_files = image_files
        self._validate()

    def image_generator(self):
        '''void->ndarray
        ndarray generator for all valid images read from the
        _image_paths list, which is set in class initialisation

        This also clears the current valid and invalid image lists and
        resets them based on if the image was successfully loaded by _cv2.imread
        '''
        self._invalid_image_files = []
        self._valid_image_files = []
        for image in self._image_files:
            if _os.path.isfile(image):
                try:
                    nd = _cv2.imread(image)
                    self._valid_image_files.append(image)
                    yield nd
                except Exception:
                    self._invalid_image_files.append(image)
            else:
                self._invalid_image_files.append(image)


class _ReadFiles(object):
    '''executes the sql and generates and instantiates an ImageValidator member
    Handles conversion of digikam stored paths to actual file paths if in
    Linux or windows

    The SQL SELECT must include the fields:
        identifier, specificPath, relativePath and name
    '''

    def __init__(self, sql, dbfile):
        '''(str, str)
        SQL is an sql statement, which must include the fields
        identifier, specificPath, relativePath and name
        '''
        self.sql = sql
        self.dbfile = dbfile
        self.images = _ImagesValidator()
        self._read()

    def read(self, sql, dbfile):
        '''refresh images read from db using a new sql
        '''
        self.sql = sql
        self.dbfile = dbfile
        if sql != '' or sql is not None:
            self._read()

    def _read(self):
        '''void->void
        Loads image pointers into the ImagesValidator object, self.images

        Requires that the initial sql contains the fields:
          identifier
          specificPath
          relativePath
          name
        '''
        image_paths = []

        with _sqlitelib.Conn(self.dbfile) as cn:
            assert isinstance(cn, _sqlite3.Connection)

            cur = cn.cursor()
            cur.execute(self.sql)
            row = cur.fetchall()
            if not SILENT:
                print('Reading image paths from digikam database')
            cnt = 0
            strip = ['-', 'VOLUMEID:?UUID=']
            drives = _get_available_drive_uuids(strip=strip)
            invalid_uuids = []
            for res in row:
                if SPECIFIC_PATH_OVERRIDE == '':
                    if _get_platform() == 'windows':
                        uuid = res['identifier'].upper()

                        for char in strip:
                            uuid = uuid.replace(char, '')

                        if uuid in drives:
                            drive = drives[uuid] + _os.sep
                        else:
                            if uuid not in invalid_uuids:
                                invalid_uuids.append(uuid)
                                _warnings.warn('No drive found for UUID', uuid)
                            drive = ''
                    elif _get_platform() == 'linux':
                        drive = ''
                    else:
                        _warnings.warn(
                            'Warning: Platform not recognised or unsupported. Treating as Linux.')
                        drive = ''
                else:
                    drive = SPECIFIC_PATH_OVERRIDE

                spath = read_specific_path(res['specificPath'])
                rpath = res['relativePath']
                name = _os.sep + res['name']
                full_path = _os.path.normpath(drive + spath + rpath + name)
                image_paths.append(full_path)
                cnt += 1
                if not SILENT:
                    _print_progress(cnt, len(row), bar_length=30)

            self.images.image_files = image_paths


class MeasuredImages(object):
    '''Gets a list of my measured images from my digikamlib
    providing an ndarray (image) generator to access the list and
    as lists to valid, invalid images and the full list of all images in
    the database.

    The list is created from a view in the digikamlib.
    '''

    def __init__(self, dbfile, digikam_measured_tag, digikam_camera_tag):
        '''initialise the class'''
        self.digikam_measured_tag = digikam_measured_tag
        self.digikam_camera_tag = digikam_camera_tag
        self.dbfile = dbfile
        self._get_measured_images()
        self.Files = None

    def _get_measured_images(self):
        '''void->list
        returns list of full image paths
        from a digikam database using the module level _CONNECTION
        which needs to be previousl set
        '''
        sql = (
            'select distinct Images.id, AlbumRoots.identifier, AlbumRoots.specificPath, Albums.relativePath, images.name '
            'from Images '
            ' inner join Albums on albums.id=images.album '
            ' inner join AlbumRoots on AlbumRoots.id=Albums.albumRoot '
            'where '
            'Images.id in ( '
            'select '
            'Images.id '
            'from images '
            'inner join ImageTags on images.id=ImageTags.imageid '
            'inner join Tags on Tags.id=ImageTags.tagid '
            'where '
            'Tags.name="camera") '
            'AND '
            'Images.id in ( '
            'select Images.id '
            'from images '
            'inner join ImageTags on images.id=ImageTags.imageid '
            'inner join Tags on Tags.id=ImageTags.tagid '
            'where '
            'Tags.name="' + self.digikam_measured_tag + '") '
            'AND '
            'Images.id in ( '
            'select Images.id '
            'from images '
            'inner join ImageTags on images.id=ImageTags.imageid '
            'inner join Tags on Tags.id=ImageTags.tagid '
            'where Tags.name="' + self.digikam_camera_tag + '")')

        self.Files = _ReadFiles(sql, self.dbfile)


class ImagePaths(object):
    '''This creates lists of paths to image files (ie the full file name, like C:/imgs/myimg.jpg)
    It also only uses files from digikam which exist on the file system.

    Create multiple ImagePaths and append results to master list
    to reproduce an OR query.
    '''

    def __init__(self, digikam_path):
        '''(str)
        set the path to the digikam db digikam4.db'''
        self.digikam_path = digikam_path


    def images_by_tags_outerAnd_innerOr(
            self,
            filename='',
            album_label='',
            relative_path='',
            **kwargs):
        '''(str, str, str,str, Key-value kwargs representing parent tag name and child tag name)->list

        Performs and between parent tags, and OR within parent tags.
        eg. (species='bass' or 'pollack') and (occlusion=0 or occlusion=10)

        Retrieve images by the tags, where kwargs is parent key=child key

        Pass in __any__ = <tag> for a single non-heirarchy match, eg __any__ = 'bass' will pick up anything tagged with bass anywhere

        Where a parent has multiple children, need to use a list for the values, e.g. species=['bass','mackerel']

        Relative paths must use forward slash /
        '''

        aliases = 'abcdefghijklmnopqrstyvwxyz'

        start_sql = 'SELECT distinct zz.id, zz.identifier, zz.specificPath, zz.relativePath, zz.name FROM '


        #now need to construct inners
        inner_template = '( select distinct ' \
            'Images.id, AlbumRoots.identifier, AlbumRoots.specificPath, Albums.relativePath, images.name ' \
            'from Images ' \
            'inner join Albums on albums.id=images.album ' \
            'inner join AlbumRoots on AlbumRoots.id=Albums.albumRoot ' \
            'where Images.id in ' \
            '( ' \
            'select images.id ' \
            'from images ' \
            'inner join imagetags on images.id=ImageTags.imageid ' \
            'inner join tags on tags.id=imagetags.tagid ' \
            'where ' \
            'imagetags.tagid in ' \
            '( ' \
            'select children.id as tagid ' \
            'from tags children ' \
            'left join tags parent on children.pid=parent.id ' \
            'where ___REPLACE___))) as _TABLE_ALIAS INNER JOIN '

        i = 0
        aliases_used = []
        lst_tables = []
        where = []

        if kwargs:
            first = True
            for key, value in kwargs.items():
                where = []

                if isinstance(value, list): #occlusion=['0', '10', '20']
                    for val in value:
                        where.append(
                            " (parent.name='%s' AND children.name='%s') OR" % (key, val))
                else: #occlusion='0'
                    where.append(" (parent.name='%s' AND children.name='%s') AND" % (key, value))

                if where[-1][-3:] == 'AND' or where[-1][-4:] == 'AND ':
                    where[-1] = _rreplace(where[-1], 'AND', '', 1)

                if where[-1][-2:] == 'OR' or where[-1][-3:] == 'OR ':
                    where[-1] = _rreplace(where[-1], 'OR', '', 1)

                w = ''.join(where)
                tmp = inner_template.replace('___REPLACE___', w)
                if first:
                    tmp = tmp.replace('_TABLE_ALIAS', aliases[i] + '\n') #as x    - ie the first table
                    first = False
                else:
                    s = aliases[i] + ' on ' + aliases[i-1] + '.id=' + aliases[i] + '.id\n' #as y on x.id=y.id
                    tmp = tmp.replace('_TABLE_ALIAS', s)

                tmp = tmp.replace("parent.name='__any__' AND", " ") #passing in _any_=['10','20] should ignore parent

                lst_tables.append(tmp)
                aliases_used.append(aliases[i])
                i += 1
            tables_sql = ''.join(lst_tables)
        else:
            tables_sql = ''





        #make the final sql query (table alias zz)
        #to filter for top level stuff, albumroot, image name etc
        lst_final_sql = ['(select distinct '
            'Images.id, AlbumRoots.identifier, AlbumRoots.specificPath, Albums.relativePath, images.name '
            'from '
            'Images '
            'inner join Albums on albums.id=images.album '
            ' inner join AlbumRoots on AlbumRoots.id=Albums.albumRoot '
            'where 1=1 '
            ]


        if filename != '':
            lst_final_sql.append(
                " AND Images.name='%s'" %
                filename.lstrip('\\').lstrip('/'))

        if relative_path != '':
            lst_final_sql.append(
                " AND Albums.relativePath='%s'" %
                _add_left(
                    relative_path.rstrip('/').rstrip('\\'),
                    '/'))  # e.g. should be /scraped/0b

        if album_label != '':
            lst_final_sql.append(
                " AND AlbumRoots.label='%s'" %
                album_label)  # album name

        if kwargs:
            s = ') as zz on ' + aliases[i-1] + '.id=zz.id' #as zz on x.id = zz.id
            lst_final_sql.append(s)
        else:
            lst_final_sql.append(') as zz')

        final_sql = ''.join(lst_final_sql)

        query = ''.join([start_sql, tables_sql, final_sql])


        rf = _ReadFiles(query, self.digikam_path)
        if not SILENT:
            print(
                'Total image paths: %s | Valid: %s | Invalid: %s' %
                (rf.images.total_count,
                 rf.images.valid_count,
                 rf.images.invalid_count))
        return rf.images.valid


    def images_by_tags_or(
            self,
            filename='',
            album_label='',
            relative_path='',
            **kwargs):
        '''(str, str, str,str, Key-value kwargs representing parent tag name and child tag name)->list

        Retrieve images by the tags, where kwargs is parent key=child key
        Hence for bass, we would call using:
          ImagesByTags(species='bass')

        Pass in __any__ = <tag> for a single non-heirarchy match, eg __any__ = 'bass' will pick up anything tagged with bass anywhere

        Where a parent has multiple children, need to use a list for the values, e.g. species=['bass','mackerel']

        Get a list of a images by the tags passed in args

        Relative paths must use forward slash /

        *Note sets can be used to emulate AND/OR/NOT like operations on returned lists
        '''
        # Albumroots.specificpath is the root path of the album from the drive, excluding the drive
        # e.g. /development/python/opencvlib/calibration for the calibration album
        # Albums.relative path is the path after the album root
        # e.g.

        sql = [
            'select distinct '
            'Images.id, AlbumRoots.identifier, AlbumRoots.specificPath, Albums.relativePath, images.name '
            'from '
            'Images '
            'inner join Albums on albums.id=images.album '
            ' inner join AlbumRoots on AlbumRoots.id=Albums.albumRoot '
            'where '
            'Images.id in '
            '( '
            'select '
            'images.id '
            ' from '
            'images '
            ' inner join imagetags on images.id=ImageTags.imageid '
            ' inner join tags on tags.id=imagetags.tagid '
            'where '
            'imagetags.tagid in '
            '( '
            'select '
            'children.id as tagid '
            'from '
            'tags children '
            'left join tags parent on children.pid=parent.id '
            'where '
        ]
        where = []
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    for val in value:
                        where.append(
                            " (parent.name='%s' AND children.name='%s') OR" % (key, val))
                else:
                    where.append(
                        " (parent.name='%s' AND children.name='%s') OR" % (key, value))

            #where = [" parent.name='%s' OR children.name='%s' OR" % (key, value) for key, value in kwargs.items()]
            where[-1] = _rreplace(where[-1], 'OR', '', 1)
        else:
            where = [" 1=1 "]

        sql.append(''.join(where))
        sql.append('))')

        sql.append(" AND 1=1 ")

        if filename != '':
            sql.append(
                " AND Images.name='%s'" %
                filename.lstrip('\\').lstrip('/'))

        if relative_path != '':
            sql.append(
                " AND Albums.relativePath='%s'" %
                _add_left(
                    relative_path.rstrip('/').rstrip('\\'),
                    '/'))  # e.g. should be /scraped/0b

        if album_label != '':
            sql.append(
                " AND AlbumRoots.label='%s'" %
                album_label)  # album name

        query = "".join(sql)

        # Fix string if we arnt bothered about a parent match for any kwarg args
        # which would look like:
        # where parent.name='__any__' AND children.name='bass'
        query = query.replace("parent.name='__any__' AND", " ")
        rf = _ReadFiles(query, self.digikam_path)

        if not SILENT:
            print(
                'Total image paths: %s | Valid: %s | Invalid: %s' %
                (rf.images.total_count,
                 rf.images.valid_count,
                 rf.images.invalid_count))
        return rf.images.valid


    #TODO Doesnt currently work because of the
    #kwarg handling
    #    (parent.name = 'fins' AND
                   #           children.name = 'dorsal_spiny')
    #        AND
        #    (parent.name = 'fins' AND
                   #           children.name = 'causal')
    #All of these cant be true
    #need to convert it to be like the OR
    def images_by_tags_and(
            self,
            filename='',
            album_label='',
            relative_path='',
            **kwargs):
        '''(str, str, str,str, Key-value kwargs representing parent tag name and child tag name)->list

        Retrieve images by the tags, where kwargs is parent key=child key
        Hence for bass, we would call using:
          ImagesByTags(species='bass')

        Pass in __any__ = <tag> for a single non-heirarchy match, eg __any__ = 'bass' will pick up anything tagged with bass anywhere

        Where a parent has multiple children, need to use a list for the values, e.g. species=['bass','mackerel']

        Get a list of a images by the tags passed in args

        Relative paths must use forward slash /

        *Note sets can be used to emulate AND/OR/NOT like operations on returned lists
        '''
        # Albumroots.specificpath is the root path of the album from the drive, excluding the drive
        # e.g. /development/python/opencvlib/calibration for the calibration album
        # Albums.relative path is the path after the album root
        # e.g.

        sql = [
            'select distinct '
            'Images.id, AlbumRoots.identifier, AlbumRoots.specificPath, Albums.relativePath, images.name '
            'from '
            'Images '
            'inner join Albums on albums.id=images.album '
            ' inner join AlbumRoots on AlbumRoots.id=Albums.albumRoot '
            'where '
            'Images.id in '
            '( '
            'select '
            'images.id '
            ' from '
            'images '
            ' inner join imagetags on images.id=ImageTags.imageid '
            ' inner join tags on tags.id=imagetags.tagid '
            'where '
            'imagetags.tagid in '
            '( '
            'select '
            'children.id as tagid '
            'from '
            'tags children '
            'left join tags parent on children.pid=parent.id '
            'where '
        ]
        where = []
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, list):
                    for val in value:
                        where.append(
                            " (parent.name='%s' AND children.name='%s') AND" % (key, val))
                else:
                    where.append(
                        " (parent.name='%s' AND children.name='%s') AND" % (key, value))

            #where = [" parent.name='%s' OR children.name='%s' OR" % (key, value) for key, value in kwargs.items()]
            where[-1] = _rreplace(where[-1], 'AND', '', 1)
        else:
            where = [" 1=1 "]

        sql.append(''.join(where))
        sql.append('))')

        sql.append(" AND 1=1 ")

        if filename != '':
            sql.append(
                " AND Images.name='%s'" %
                filename.lstrip('\\').lstrip('/'))

        if relative_path != '':
            sql.append(
                " AND Albums.relativePath='%s'" %
                _add_left(
                    relative_path.rstrip('/').rstrip('\\'),
                    '/'))  # e.g. should be /scraped/0b

        if album_label != '':
            sql.append(
                " AND AlbumRoots.label='%s'" %
                album_label)  # album name

        query = "".join(sql)

        # Fix string if we arnt bothered about a parent match for any kwarg args
        # which would look like:
        # where parent.name='__any__' AND children.name='bass'
        query = query.replace("parent.name='__any__' AND ", " ")
        rf = _ReadFiles(query, self.digikam_path)

        if not SILENT:
            print(
                'Total image paths: %s | Valid: %s | Invalid: %s' %
                (rf.images.total_count,
                 rf.images.valid_count,
                 rf.images.invalid_count))
        return rf.images.valid
