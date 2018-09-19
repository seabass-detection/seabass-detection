# pylint: disable=C0103, locally-disabled, attribute-defined-outside-init, protected-access, unused-import, arguments-differ, unused-argument
#TOO Suppress excepions globally for this module when in SILENT mode, check iolib which has the routine to do it
'''Image generatrs for multiple sources.

All yielded generators have an error handler wrapping which logs errors
to prevent stop failures during big processing tasks

NEW GENERATORS
    Filter (sieve) and Transform Support
        When providing new generators, ensure to add delegate filtering
        and transformation by adding the following after the image is obtained:

        img = _cv2.imread(fname, outflag)
        img = super().generate(img) #transforms and checks the filter
    Yield
        All generators should yield ndarray, filepath, dict
        Where dict is generator specific information.
    '''

from os import path as _path
import abc as _abc
from enum import Enum as _Enum
from warnings import warn as _warn
import logging as _logging
from inspect import getsourcefile as _getsourcefile

import cv2 as _cv2
import numpy as _np
from numpy.random import randint as _randint


import funclib.baselib as _baselib
import funclib.iolib as _iolib

import opencvlib.decs as _decs
from opencvlib.info import ImageInfo as _ImageInfo
from opencvlib import getimg as _getimg

from opencvlib.distance import nearN_euclidean as _nearN_euclidean

import opencvlib.imgpipes.vgg as _vgg
import opencvlib.imgpipes.digikamlib as _digikamlib
import opencvlib.imgpipes.filters as _filters
from opencvlib.imgpipes import config as _config
import opencvlib.imgpipes.voc_utils as _voc

import opencvlib.roi as _roi
import opencvlib.transforms as _transforms
from opencvlib import IMAGE_EXTENSIONS_AS_WILDCARDS as _IMAGE_EXTENSIONS_AS_WILDCARDS

#__all__ = ['image_generator', 'Images', 'DigikamSearchParams', 'VGGSearchParams']

# hard coded vgg file which needs to be in each directory
VGG_FILE = 'vgg.json'


SILENT = True

_pth = _iolib.get_file_parts2(_path.abspath(_getsourcefile(lambda: 0)))[0]
_LOGPATH = _path.normpath(_path.join(_pth, 'features.py.log'))
_logging.basicConfig(format='%(asctime)s %(message)s', filename=_LOGPATH, filemode='w', level=_logging.DEBUG)


def _prints(s, log=True):
    '''silent print'''
    if not SILENT:
        print(s)
    if log:
        _logging.propagate = False
        _logging.info(s)
        _logging.propagate = True

_prints('Logging to %s' % _LOGPATH)




class eDigiKamSearchType(_Enum):
    '''digikam search tpye'''
    and_ = 0
    or_ = 1
    innerOr_outerAnd = 2


#region Generator related classes
class _BaseGenerator(_abc.ABC):
    '''abstract class for all these generator functions
    '''
    @_abc.abstractmethod
    def generate(self):
        '''placeholder'''
        pass


#TODO Implement regex sieve on image filenames
#Do it in the base class so it is used by all
#classes which inherit from _Generator
class _Generator(_BaseGenerator):
    '''base generator class
    Use with BaseGenerator to create new generators

    Pops transforms and filters.

    When instantiating classes which inherit Generator
    provide kwargs with
    transforms=Transforms (Transforms class - a collection of Transfor classes)
    filters=Filters (Filters class - a collection of Filter classes)
    '''
    def __init__(self, *args, **kwargs):
        self._transforms = kwargs.pop('transforms', None)
        if not isinstance(self._transforms, _transforms.Transforms) and not self._transforms is None:
            raise ValueError('Base generator class keyword argument "transforms" requires class type "transforms.Transforms"')

        self._filters = kwargs.pop('filters', None)
        if not isinstance(self._filters, _filters.Filters) and not self._filters is None:
            raise ValueError('Base generator class keyword argument "filters" requires class type "filters.Filters"')
       # assert isinstance(self._transforms, _transforms.Transforms)
       # assert isinstance(self._filters, _filters.Filters)


    @property
    def transforms(self):
        '''transforms getter'''
        return self._transforms
    @transforms.setter
    def transforms(self, transforms):
        '''transforms setter'''
        self._transforms = transforms


    @property
    def filters(self):
        '''filters getter'''
        return self._filters
    @filters.setter
    def filters(self, filters):
        '''filters setter'''
        self._filters = filters


    @_decs.decgetimgmethod
    def executeTransforms(self, img):
        '''(ndarray|str)->ndarray
        execute transforms enqueued in the Transforms class
        and return the transformed image
        '''
        #img = _getimg(img)
        if isinstance(self._transforms, _transforms.Transforms):
            try:
                img = self.transforms.executeQueue(img)
            except ValueError as e:
                _logging.exception(str(e))
        return img


    @_decs.decgetimgmethod
    def isimagevalid(self, img):
        '''does the image pass a filter
        '''
        #img = _getimg(img)
        #assert isinstance(self.filters, _filters.Filters)
        if isinstance(self.filters, _filters.Filters):
            return self.filters.validate(img)

        return True


    @_decs.decgetimgmethod
    def generate(self, img):
        '''(str|ndarray,cv2.imread flag..)->None|ndarray
        takes img, applies relvant filters and transforms
        and returns to calling generater to bubble up the
        transformed image

        Returns None if filter fails
        '''
        if self.isimagevalid(img):
            img = self.executeTransforms(img)
            return img

        return None


#endregion



#region Params VGG and Digikam Filter Classes
class VGGSearchParams(object):
    '''VGGSearchParams
    '''
    def __init__(self, folders, parts, species, recurse=False):
        '''(list|str, list|str, list|str, bool)
        folders: paths to folders containing vgg.json regions file and associated images
        parts: region name tags, such as head or whole
        species: species
        recurse: recurse down directories in folders. Only folders with a vgg.json are relevant
        '''
        if isinstance(parts, str):
            parts = [parts]

        if isinstance(species, str):
            species = [species]

        if isinstance(folders, str):
            folders = [folders]

        self._parts = _vgg._valid_parts_in_list(parts)
        self._species = _vgg._valid_species_in_list(species)
        self._folders = folders
        self.recurse = recurse

    @property
    def parts(self):
        '''regions getter'''
        return self._parts

    @parts.setter
    def parts(self, parts):
        '''regions setter'''
        self._parts = _vgg._valid_parts_in_list(parts)

    @property
    def species(self):
        '''species getter'''
        return self._species

    @species.setter
    def species(self, species):
        '''species setter'''
        self._species = _vgg._valid_species_in_list(species)

    @property
    def folders(self):
        '''folders getter'''
        return self._folders
    @folders.setter
    def folders(self, folders):
        '''folders setter'''
        self._folders = [folders] if isinstance(folders, str) else folders



class DigikamSearchParams():
    '''digikam search params
    filename:
        e.g. 12345.jpg
    album_label:
        The album name
        SQLLite:AlbumRoots.label
        e.g. images, calibration
    relative_path:
        The albums relative folder path on the file system
        SQLLite:Albums.relativePath
        e.g. /bass/angler
    search_type:
        booling rules to apply
    keyvaluetags:
        An unlimited number of key/value tags, where the key
        is the parent tag and the values are subtags, noting subtags
        can be a list.
        Should support the special key value _any_, which will ignore
        the parent.
        Examples:
            __any__ = ['tag1', 'tag2']
            parenttag1=['tag1', 'tag2], parenttag2=['tag1', 'tag2']
    '''
    SearchType = eDigiKamSearchType

    def __init__(self, filename='', album_label='', relative_path='', search_type=eDigiKamSearchType.innerOr_outerAnd, **keyvaluetags):
        self.filename = filename
        self.album_label = album_label
        self.relative_path = relative_path
        self.search_type = search_type
        self.keyvaluetags = keyvaluetags

    def no_filters(self):
        '''()->bool
        returns true if no filters set
        '''
        return self.filename == '' and self.album_label == '' and self.relative_path == '' and not self.keyvaluetags
#endregion



#region Generators
class DigiKam(_Generator):
    '''Generate images based on a digikam filter.
    Instantiate with an instance of generators.DigikamSearchParams.

    Parameters:
        digikam_params:
            an instance of DigikamSearchParams

    Methods
        generate:generate images, or just paths

    Comment:
        See DigikamSearchParams.__doc__ for parameter options.
    '''

    #load db in class context for efficiency
    dkImages = _digikamlib.ImagePaths(_config.digikamdb)

    def __init__(self, digikam_params, *args, **kwargs): #args and kwargs get passed back to the base class - supports filters and transforms
        #super call to base supports passing kawrgs for
        #transforms and filters
        self._digikam_params = digikam_params
        assert isinstance(self._digikam_params, DigikamSearchParams)
        self._dirty_filters = True
        self._image_list = None
        self._refresh_image_list()
        super().__init__(*args, **kwargs)


    @property
    def image_list(self):
        '''() -> list
        Returns the list of image paths
        '''
        return self._image_list


    @property
    def digikam_params(self):
        '''digikam_params getter'''
        return self._digikam_params
    @digikam_params.setter
    def digikam_params(self, digikam_params):
        '''digikam_params setter'''
        self._digikam_params = digikam_params
        self._dirty_filters = True
        self._refresh_image_list()


    def generate(self, yield_path_only=False, outflag=_cv2.IMREAD_UNCHANGED):
        '''(bool)->ndarray,str
        generate whole images applying only the digikam filter
        Yields:
        image region (ndarray), full image path (eg c:/images/myimage.jpg)

        If yeild_path_only, then the yielded ndarray will be none.

        Has an error handler which logs failures in the generator
        '''

        if self._digikam_params.no_filters() or not self.image_list:
            return

        for imgpath in self.image_list:
            try:
                if yield_path_only:
                    yield None, imgpath, {}
                else:
                    img = _getimg(imgpath, outflag)
                    img = super().generate(img) #Check filters an execute transforms
                    if isinstance(img, _np.ndarray):
                        yield img, imgpath, {}
            except Exception as dummy:
                s = 'Processing of %s failed.' % imgpath
                _logging.exception(s)


    def _refresh_image_list(self, force=False):
        '''(bool)->list
        force:
            force reload of the image list even if filters are not dirty

        Loads list of digikam images according to the digikamsearch parameters
        '''

        if self.digikam_params.no_filters():
            self.image_list = []
            return None

        assert isinstance(self.digikam_params, DigikamSearchParams)
        assert DigiKam.dkImages is not None
        if force or self._dirty_filters:
            if self.digikam_params.search_type == eDigiKamSearchType.and_:
                imgs = DigiKam.dkImages.images_by_tags_and(
                    filename=self.digikam_params.filename,
                    album_label=self.digikam_params.album_label,
                    relative_path=self.digikam_params.relative_path,
                    **self.digikam_params.keyvaluetags
                        )
            elif self.digikam_params.search_type == eDigiKamSearchType.or_:
                imgs = DigiKam.dkImages.images_by_tags_or(
                    filename=self.digikam_params.filename,
                    album_label=self.digikam_params.album_label,
                    relative_path=self.digikam_params.relative_path,
                    **self.digikam_params.keyvaluetags
                        )
            elif self.digikam_params.search_type == eDigiKamSearchType.innerOr_outerAnd:
                imgs = DigiKam.dkImages.images_by_tags_outerAnd_innerOr(
                    filename=self.digikam_params.filename,
                    album_label=self.digikam_params.album_label,
                    relative_path=self.digikam_params.relative_path,
                    **self.digikam_params.keyvaluetags
                        )
            else:
                raise UserWarning('Invalid DigikamSearchParams search type. Search type should be Enum:eDigiKamSearchType')

            self._image_list = [_iolib.fixp(i) for i in imgs]



class FromPaths(_Generator):
    '''Generate images from a list of folders
    Transforms and filters can be added by instantiating lists Transform and Filter
    objects and passing them as named arguments. See test_generators for more
    examples.

    paths:
        Single path or list/tuple of paths
    wildcards:
        Single file extension or list of file extensions.
        Extensions should be dotted, an asterix is appended
        if none exists.

    Yields: ndarray, str, dict   i.e. the image, image path, {}

    Example:
        fp = generators.FromPaths('C:/temp', wildcards='*.jpg',
                            transforms=Transforms, filters=Filters)
    '''
    def __init__(self, paths, *args, wildcards=_IMAGE_EXTENSIONS_AS_WILDCARDS, **kwargs):
        self._paths = paths
        self._wildcards = wildcards
        super().__init__(*args, **kwargs)


    @property
    def paths(self):
        '''paths getter'''
        return self._paths
    @paths.setter
    def paths(self, paths):
        '''paths setter'''
        self._paths = paths


    def generate(self, outflag=_cv2.IMREAD_UNCHANGED, pathonly=False, recurse=False):
        '''(cv2.imread option, bool, bool) -> ndarray, str, dict
        Globs through every file in paths matching wildcards returning
        the image as an ndarray

        recurse:
            Recurse through paths
        outflag:
            <0 - Loads as is, with alpha channel if present)
            0 - Force grayscale
            >0 - 3 channel color iage (stripping alpha if present
        pathonly:
            only generate image paths, the ndarray will be None

        Yields:
            image, path, an empty dictionary

        Notes:
            The empty dictionary is yielded so it is the same format as other generators
         '''

        for imgpath in _iolib.file_list_generator1(self._paths, self._wildcards, recurse=recurse):
            try:
                if pathonly:
                    yield None, imgpath, {}
                else:
                    img = _cv2.imread(imgpath, outflag)
                    img = super().generate(img) #delegate to base method to transform and filter (if specified)
                    if isinstance(img, _np.ndarray):
                        yield img, imgpath, {}
            except Exception as dummy:
                s = 'Processing of %s failed.' % imgpath
                _logging.exception(s)



class VGGDigiKam(_Generator):
    '''Generate regions configured in VGG
    Example:
        VGGRegions(self.dkPos, self.vggPos, filters=None, transforms=None)
    '''
    def __init__(self, digikam_params, vgg_params, *args, **kwargs):
        '''(DigikamSearchParams, VGGSearchParams, bool)->yields ndarray
        folders must have a vgg.json file in them, otherwise they will be ignored
        '''
        self.silent = False
        self._digikam_params = digikam_params
        self._vgg_params = vgg_params
        self._dirty_filters = True
        super().__init__(*args, **kwargs)

    @property
    def digikamParams(self):
        '''digikamParams getter'''
        return self._digikam_params
    @digikamParams.setter
    def digikamParams(self, digikamParams):
        '''digikamParams setter'''
        self._digikam_params = digikamParams
        self.dirty_filters = True #for use in inherited classes, not needed for this explicitly


    @property
    def vggParams(self):
        '''vggParams getter'''
        return self._vgg_params
    @vggParams.setter
    def vggParams(self, vggParams):
        '''vggParams setter'''
        self._vgg_params = vggParams
        self.dirty_filters = True #for use in inherited classes, not needed for this explicitly


    def generate(self, *args, pathonly=False, outflag=_cv2.IMREAD_UNCHANGED, file_attr_match=None, **kwargs):
        '''(bool, int, dict)-> ndarray,str, dict

        Uses the filters set in the VGGFilter and DigikamSearchParams
        to yield image regions to the caller.

        pathonly:
            Only yield the image path, None will be returned for the ndarray
        outflag:
            cv2 flag for imread
            cv2.IMREAD_COLOR|cv2.IMREAD_GRAYSCALE|cv2.IMREAD_UNCHANGED
        file_attr_match:
            A dictionary, Only generate regions for file which match these attributes.
            e.g. file_attr_match = {'train'=1}

        Returns [Yields]
            image region, imgpath, dictionary containing additional infornation
            Dictionary output is: {'species':spp, 'part':part, 'shape':region.shape, 'mask':mask, 'roi':region.all_points
        '''
        if self.digikamParams is None:
            dk_image_list = []
        else:
            if self.digikamParams.no_filters():
                dk_image_list = []
            else:
                dkGen = DigiKam(self.digikamParams)
                dk_image_list = [i for unused_, i, dummy in dkGen.generate(yield_path_only=True)]

        if self.vggParams.recurse:
            for fld in _iolib.folder_generator(self.vggParams.folders):

                try:
                    if _dir_has_vgg(fld):
                        if fld.endswith(VGG_FILE):
                            p = _iolib.fixp(fld)
                        else:
                            p = _iolib.fixp(_path.join(fld, VGG_FILE))

                        _vgg.load_json(p)
                        if not self.silent:
                            print('Opened regions file %s' % p)

                        for Img in _vgg.imagesGenerator(file_attr_match=file_attr_match):

                            if dk_image_list:
                                if not Img.filepath in dk_image_list:  # effectively applying a filter for the digikamlib conditions
                                    continue

                            for spp in self.vggParams.species:
                                for subject in Img.subjects_generator(spp):
                                    assert isinstance(subject, _vgg.Subject)
                                    for part in self.vggParams.parts:  # try all parts, eg whole, head
                                        for region in subject.regions_generator(part):
                                            assert isinstance(region, _vgg.Region)
                                            if pathonly:
                                                cropped_image = None
                                                mask = None
                                            else:
                                                i = _getimg(Img.filepath, outflag)
                                                i = super().generate(i)

                                                if isinstance(i, _np.ndarray):
                                                    mask, dummy, dummy1, cropped_image = _roi.roi_polygons_get(i, region.all_points) #3 is the image cropped to a rectangle, with black outside the region
                                                else:
                                                    _logging.warning('File %s was readable, but ignored because of a filter or failed image transformation. This can usually be ignored.')
                                                    cropped_image = None

                                            yield cropped_image, Img.filepath, {'species':spp, 'part':part, 'shape':region.shape, 'mask':mask, 'roi':region.all_points}

                except Exception as dummy:
                    s = 'Processing of file:%s failed.' % Img.filepath
                    _logging.exception(s)
        else:
            for fld in self.vggParams.folders:

                try:
                    if _dir_has_vgg(fld):
                        if fld.endswith(VGG_FILE):
                            p = _iolib.fixp(fld)
                        else:
                            p = _iolib.fixp(_path.join(fld, VGG_FILE))
                        _vgg.load_json(p)
                        if not self.silent:
                            print('Opened regions file %s' % p)

                        for Img in _vgg.imagesGenerator(file_attr_match=file_attr_match):

                            if dk_image_list:
                                if not Img.filepath in dk_image_list:  # effectively applying a filter for the digikamlib conditions
                                    continue
                                else:
                                    #z = 1
                                    pass

                            for spp in self.vggParams.species:
                                for subject in Img.subjects_generator(spp):
                                    assert isinstance(subject, _vgg.Subject)
                                    for part in self.vggParams.parts:
                                        for region in subject.regions_generator(part):
                                            assert isinstance(region, _vgg.Region)
                                            if pathonly:
                                                cropped_image = None
                                                mask = None
                                            else:
                                                i = _getimg(Img.filepath, outflag)
                                                i = super().generate(i)

                                                if isinstance(i, _np.ndarray):
                                                    mask, dummy, dummy1, cropped_image = _roi.roi_polygons_get(i, region.all_points) #3 is the image cropped to a rectangle, with black outside the region
                                                else:
                                                    _logging.warning('File %s was readable, but ignored because of a filter or failed image transformation. This can usually be ignored.')
                                                    cropped_image = None

                                            yield cropped_image, Img.filepath, {'species':spp, 'part':part, 'shape':region.shape, 'mask':mask, 'roi':region.all_points}
                except Exception as dummy:
                    s = 'Processing of file:%s failed.' % Img.filepath
                    _logging.exception(s)


class VGGImages(_Generator):
    '''Generate images only specified in a VGG file.
    This does not generate ROIs.

    vgg_file_paths:
        Full paths to vgg files, e.g.
        'c:/vgg.json'
        ['c:/vgg.json','c:/vgg_other.json']

    filters:
        Keyword argument containing a list of filter function from imgpipes.filters, passed as filters=...
    transforms:
        Keyword argument containing lit of ist of imgpipes.transforms functions, passed as transforms=...
        to apply to output image

    Example:
        gen = VGGImages('c:/vgg.json')
        for img, filepath, d in gen()
            #do work

    See test/test_generators.py for some examples.
    '''

    def __init__(self, vgg_file_paths, *args, **kwargs):
        '''(str|list, list|None, dict|None) -> void'''
        if isinstance(vgg_file_paths, str):
            vgg_file_paths = [vgg_file_paths]
        self.vgg_file_paths = vgg_file_paths
        '''Paths to vgg json files, e.g. 'C:/vgg.json'''
        self.silent = True
        '''Suppress console messages'''

        super().__init__(*args, **kwargs)


    def generate(self, path_only=False, outflag=_cv2.IMREAD_UNCHANGED, file_attr_match=None):
        '''(bool, int, dict|None) -> ndarray|None, str, dict
        Generate images and paths, dict will be empty.

        file_attr_match:
            a dictionary which is checked for a partial
            match against the image file attributes,
            eg {'is_train'=1}

        Example:
        >>>for Img, pth, _ in VGGImages.generate(file_attr_match={'train'=1}):
            #do some work
        '''
        for vgg_file in self.vgg_file_paths:
            try:
                _vgg.load_json(vgg_file)
                for I in _vgg.imagesGenerator(file_attr_match=file_attr_match):
                    if path_only:
                        yield None, I.filepath, None
                        continue
                    else:
                        img = _cv2.imread(I.filepath, flags=outflag)
                        img = super().generate(img)
                        yield img, I.filepath, {}
            except Exception as e:
                _logging.exception(e)
                if not self.silent:
                    _warn(str(e))


class VGGROI(_Generator):
    '''Generate images specified in a VGG file

    Accepts a list of shapes types to return, and
    a dictionary to check for an item by item match
    with the VGG region attributes per image.

    vgg_file_paths:
        Full paths to vgg files, e.g.
        'c:/vgg.json'
        ['c:/vgg.json','c:/vgg_other.json']

    filters:
        Keyword argument containing a list of filter function from imgpipes.filters, passed as filters=...
    transforms:
        Keyword argument containing lit of ist of imgpipes.transforms functions, passed as transforms=...
        to apply to output image
    Example:
        gen = VGG('c:/vgg.json', ['ellipse', 'rect'], {'part': 'hand', 'id': '1'})

    See test/test_generators.py for some examples.
    '''
    def __init__(self, vgg_file_paths, *args, region_attrs=None, **kwargs):
        '''(str|list, list|None, dict|None) -> void'''
        if isinstance(vgg_file_paths, str):
            vgg_file_paths = [vgg_file_paths]
        self.vgg_file_paths = vgg_file_paths
        '''Paths to vgg json files, e.g. 'C:/vgg.json'''
        self.silent = True
        '''Suppress console messages'''

        self.region_attrs = region_attrs
        super().__init__(*args, **kwargs)


    def generate(self, shape_type='rect', region_attrs=None, path_only=False, outflag=_cv2.IMREAD_UNCHANGED, skip_imghdr_check=False, grow_roi_proportion=1, grow_roi_x=1, grow_roi_y=1, file_attr_match=None):
        '''(str|list|None, dict, bool, cv2.imread option, bool, float, float, float, dict) -> ndarray|None, str, dict
        Yields the images with the bounding boxes and category name of all objects
        in the pascal voc images

        The region_attributes are yielded in the dictionary object.
        shape_type:
            The type of shape. ONLY SUPPORTS rect at the moment
            #Valid values are 'rect', 'circle', 'point', 'ellipse'.
            Also accepts a list or tuple. If None, then all shapes yielded.
        region_attr_match:
            A dictionary which is matched against the region attributes
            of the shape.
        path_only:
            Only yield the path to the image, the image and dict returned will
            be None
        outflag:
            <0  Loads as is, with alpha channel if present
            0   Force grayscale
            >0  3 channel color iage (stripping alpha if present
        skip_imghdr_check:
            The Python imghdr lib is used to check for an image, but this can
            be unreliable. simple_file_check will just check that the image
            exists.
        grow_roi_proportion:
            increase or decreaese roi by this percentage of the
            original roi. Override grow_roi_x and grow_roi_y if
           not 1.
        grow_roi_x:
            Increase or decrease roi width by this amount.
        grow_roi_y:
            Increase or decrease roi height by this amount.
        file_attr_match:
            If not None, only regions from images with the matching file attributes
            will be generated

        Yields:
            image, path, {'region_attributes':reg, 'pts_cvxy':pts_cvxy}
        or
            None, path, region_attributes dict

            Where region_attributes is the JOSN dict from the vgg file

        Notes:
            If yields a rectangle, the coordinates can be accessed from
            the returned
            dict['region_attributes'].x, .y, .w, .h

            Resized coordinates can be accessed with:
            dict['pts_grown_cvxy']

            If path_only is True and the grow options are defined, no lower or upper
            cap is set on points, i.e.they may lay outside the image.
         '''
        # TODO Support other 2D-shapes, currently only rects
        assert shape_type == 'rect', 'Only rectangles are supported'

        for vgg_file in self.vgg_file_paths:
            try:
                _vgg.load_json(vgg_file)
                for I in _vgg.imagesGenerator(skip_imghdr_check=skip_imghdr_check, file_attr_match=file_attr_match):
                    for reg in I.roi_generator(shape_type, self.region_attrs):
                        assert isinstance(reg, _vgg.Region)
                        if path_only:
                            if grow_roi_proportion != 1 or grow_roi_x != 1 or grow_roi_y != 1:
                                if grow_roi_proportion != 1:
                                    if grow_roi_proportion > 1:
                                        _warn('grow_roi_proportion was > 1, but path_only was specified. Cannot apply ceiling and floor.')
                                    pts = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                                    pts = _roi.roi_rescale(pts, grow_roi_proportion)
                                else:
                                    pts = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                                    if grow_roi_x > 1 or grow_roi_y > 1:
                                        _warn('grow_roi_x or grow_roi_y was > 1, but path_only was specified. Cannot apply ceiling and floor.')

                                    if grow_roi_x != 1:
                                        pts = _roi.roi_rescale2(pts, proportion_x=grow_roi_x)

                                    if grow_roi_y != 1:
                                        pts = _roi.roi_rescale2(pts, proportion_y=grow_roi_y)
                                    pts_cvxy = list(pts) #make a copy
                            else:
                                pts_cvxy = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                            yield None, I.filepath, {'region_attributes':reg, 'pts_grown_cvxy':pts_cvxy}
                        else:
                            img = _cv2.imread(I.filepath, flags=outflag)
                            img = super().generate(img) #filter and transform with base class
                            if img is None:
                                continue

                            if grow_roi_proportion != 1 or grow_roi_x != 1 or grow_roi_y != 1:
                                if grow_roi_proportion != 1:
                                    pts = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                                    pts = _roi.roi_rescale(pts, grow_roi_proportion, h=img.shape[0], w=img.shape[1])
                                    pts_cvxy = list(pts)
                                else:
                                    pts = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                                    if grow_roi_x != 1:
                                        pts = _roi.roi_rescale2(pts, proportion_x=grow_roi_x, h=img.shape[0], w=img.shape[1])

                                    if grow_roi_y != 1:
                                        pts = _roi.roi_rescale2(pts, proportion_y=grow_roi_y, h=img.shape[0], w=img.shape[1])
                                    pts_cvxy = list(pts) #make a copy
                                pts = _roi.rect_as_rchw(pts) #r,c,h,w
                                img, _ = _roi.cropimg_xywh(img, pts[1], pts[0], pts[3], pts[2])
                            else:
                                pts_cvxy = _roi.rect_as_points(reg.y, reg.x, reg.w, reg.h)
                                img, _ = _roi.cropimg_xywh(img, reg.x, reg.y, reg.w, reg.h)
                            yield img, I.filepath, {'region_attributes':reg, 'pts_grown_cvxy':pts_cvxy}
            except Exception as e:
                _logging.exception(e)
                if not self.silent:
                    _warn(str(e))


class RandomRegions(DigiKam):
    '''Get random regions from the digikam library
    based on search criteria which are first set
    in the DigikamSearchParams class.

    generate:
        Returns none on error
    '''
    def __init__(self, digikam_params, *args, **kwargs):
        #resolutions is a dictionary of dictionaries {'c:\pic.jpg':{'w':1024, 'h':768}}

        #Code relies on d_res and pt_res having same indices
        #All points are w,h
        self.d_res = _baselib.odict() #ordered dictionary (collections.OrderedDict) of resolutions
        self.pt_res = [] #resolutions as a list of points
        super().__init__(digikam_params, *args, **kwargs)


    def generate(self, img, region_w, region_h, sample_size=10, mask=None, outflag=_cv2.IMREAD_UNCHANGED):
        '''(ndarray|str|tuple, int, int, int, ndarray, int)->ndarray|None, str, dict
        Retrieve a random image region sampled from the digikam library
        nearest in resolution to the passed in image.

        **NOT A GENERATOR, RETURNS A SINGLE IMAGE**

        img:
            Original image to get resolution. This can be an ndarray, path (c:/pic.jpg) or list like point (w, h)
        region_w, region_h:
            Size of region to sample from similiar resolution image
        sample_size:
            Randomly sample an image from the sample_size nearest in resolution from the digikam library
        mask:
            optional white mask to apply to the image
        outflag:
            cv2.imread flag, determining the format of the returned image
            cv2.IMREAD_COLOR
            cv2.IMREAD_GRAYSCALE
            cv2.IMREAD_UNCHANGED

        Returns an image (ndarray) and the image path.

        Can return None as image.

        No error handler
        '''
        imgout = None
        try:

            if self.digikam_params.no_filters():
                raise ValueError('No digikam filters set. '
                                 'Create a class instance of generators.DigikamSearchParams and '
                                 'pass to RandomRegions.digikamParams property, or set at creation')

            self._loadres() #refresh if needed
            if isinstance(img, _np.ndarray):
                h = img.shape[0]
                w = img.shape[1]
            elif isinstance(img, (list, tuple)):
                w = img[0]
                h = img[1]
            else:
                w, h = _ImageInfo.resolution(img)

            #Now get a random sample image of about closest resolution
            samples = _nearN_euclidean((w, h), self.pt_res, sample_size)
            samples = [self.d_res.getbyindex(ind) for ind, dist in samples]

            counter = 0 #counter to break out if we are stuck in the loop
            while True:
                samp_path = samples[_randint(0, len(samples))]
                sample_image = _getimg(samp_path, outflag)[0]

                if self.isimagevalid(sample_image):
                    if region_w <= samp_path[1][0] and region_h <= samp_path[1][1]:
                        sample_image = self.executeTransforms(sample_image)
                        imgout = _roi.sample_rect(sample_image, region_w, region_h)
                        break
                    elif region_w <= samp_path[1][1] and region_h <= samp_path[1][0]: #see if ok
                        sample_image = self.executeTransforms(sample_image)
                        imgout = _roi.sample_rect(sample_image, region_h, region_w)
                        imgout = _transforms.rotate(imgout, -90, no_crop=True)
                        assert imgout.shape[0] == region_h and imgout.shape[1] == region_w
                        break
                    elif len(samples) > 1: #image too small to get region sample, delete it and try again
                        samples.remove(samp_path)
                    elif counter > 20: #lets not get in an infinite loop
                        sample_image = self.executeTransforms(sample_image)
                        imgout = sample_image
                        break
                    else: #Last one, just use it
                        sample_image = self.executeTransforms(sample_image)
                        imgout = sample_image
                        break
                counter += 1
        except Exception as e:
            print('Error was ' + str(e))
        finally:
            if imgout is None: #catch all
                return None, samp_path[0], {}

        if not mask is None:
            imgout = _roi.get_image_from_mask(imgout, mask=mask)

        return imgout, samp_path[0], {}



    @staticmethod
    def _gen_res_key(w, h):
        '''(int,int)->str
        generate unique resolution key
        '''
        return str(w) + 'x' + str(h)


    def _loadres(self, force=False):
        '''loads resolutions with full image paths
        '''
        if not force and not self._dirty_filters:
            return

        self.d_res = _baselib.odict()
        self.pt_res = []

        if self.digikam_params.no_filters():
            return

        for dummy, img_path, dummy in super().generate(yield_path_only=True):
            w, h = _ImageInfo.resolution(img_path) #use the pil lazy loader
            self.d_res[img_path] = (w, h) #ordered dict, matching order of pts, order is critical for getting random image
            self.pt_res.append([w, h])

        self._dirty_filters = False
#endregion


class RegionDualPosNeg():
    '''Generate positive and negative
    training images using VGG defined regions and
    a search specified by digikam tags.

    Negative regions are generated by using
    seperately defined digikam tags and picking randomly from
    images closest in resolution to the training image
    and deriving a randomly placed sample region of
    the same shape of the positive region
    '''
    def __init__(self, vggPos, dkPos, vggNeg, dkNeg, T, F):
        '''(VGGSearchParams, DigiKamSearchParams, VGGSearchParams, DigiKamSearchParams, Transforms, Filters)

        Generate a positive and negtive training image from
        VGG configured regions
        '''
        self.vggPos = vggPos
        self.vggNeg = vggNeg
        self.dkPos = dkPos
        self.dkNeg = dkNeg
        self.T = T
        self.F = F
        self._pos_list = []
        self._neg_list = []
        assert isinstance(self.T, _transforms.Transforms)
        assert isinstance(self.F, _filters.Filters)


    def __call__(self, vggPos, dkPos, vggNeg, dkNeg, T, F):
        '''(VGGSearchParams, DigiKamSearchParams, DigiKamSearchParams, Transforms, Filters)

        All classes in opencvlib.generators
        '''
        self.vggPos = vggPos
        self.vggNeg = vggNeg
        self.dkPos = dkPos
        self.dkNeg = dkNeg
        self.T = T
        self.F = F
        self._pos_list = []
        self._neg_list = []


    def generate(self, outflag=_cv2.IMREAD_UNCHANGED):
        '''(cv2 imread flag) -> ndarray|None, str, ndarray|None, str
        Generate train and test image regions.

        Yields:
        positive image region,
        postive image path,
        negative image region,
        negative image path

        Images return None if an error occurs
        '''
        #Get training region
        self._pos_list = []
        self._neg_list = []

        assert isinstance(self.T, _transforms.Transforms)
        assert isinstance(self.F, _filters.Filters)

        PipePos = VGGDigiKam(self.dkPos, self.vggPos, filters=None, transforms=None)
        PipeNeg = VGGDigiKam(self.dkNeg, self.vggNeg, filters=None, transforms=None)

        for dummy, img_path, argsout in PipePos.generate(pathonly=True, outflag=outflag): #Pipe is a region generator
            self._pos_list.append([img_path, argsout.get('roi')])

        for dummy, img_path, argsout in PipeNeg.generate(pathonly=True, outflag=outflag): #Pipe is a region generator
            self._neg_list.append([img_path, argsout.get('roi')])

        for ind, x in enumerate(self._pos_list):

            try:
                imgpos = _getimg(x[0])
                imgneg = _getimg(self._neg_list[ind][0])

                if not self.F.validate(imgpos): continue
                if not self.F.validate(imgneg): continue

                imgposT = self.T.executeQueue(imgpos)
                imgnegT = self.T.executeQueue(imgneg)

                if imgposT is None: continue
                if imgnegT is None: continue

                ipos = _roi.roi_polygons_get(imgposT, x[1])
                ineg = _roi.roi_polygons_get(imgnegT, self._neg_list[ind][1])
            except Exception as _:
                s = 'Failed to generate a test region for %s' % img_path
                _logging.warning(s)
                ipos = None
                ineg = None
            finally:
                yield ipos, x[0], ineg, self._neg_list[ind][0]



class RegionPosRandomNeg():
    ''' wrapper to generate a region with
    a corresponding random negative image of equal region
    size from an image of same approximate resolution.

    Train images require a VGGSearch parameter to
    extract prelabelled regions.
    '''
    def __init__(self, vggSP, pos_dkSP, neg_dkSP, T, F):
        '''(VGGSearchParams, DigiKamSearchParams, DigiKamSearchParams, Transforms, Filters)

        All classes in opencvlib.generators
        '''
        self.vggSP = vggSP
        self.pos_dkSP = pos_dkSP
        self.neg_dkSP = neg_dkSP
        self.T = T
        self.F = F


    def __call__(self, vggSP, pos_dkSP, neg_dkSP, T, F):
        '''(VGGSearchParams, DigiKamSearchParams, DigiKamSearchParams, Transforms, Filters)

        All classes in opencvlib.generators
        '''
        self.vggSP = vggSP
        self.pos_dkSP = pos_dkSP
        self.neg_dkSP = neg_dkSP
        self.T = T
        self.F = F


    def generate(self, outflag=_cv2.IMREAD_UNCHANGED):
        '''
        Generate train and test image regions.

        Returns:
            positive image region, negative image region, dict
            where dict = {'imgpath':img_path, 'region_img_path':region_img_path}
        '''
        #Get training region
        Pipe = VGGDigiKam(self.pos_dkSP, self.vggSP, filters=self.F, transforms=self.T)

        #Instantiate random sample generator class
        RR = RandomRegions(self.neg_dkSP, filters=self.F, transforms=self.T)

        for img, img_path, argsout in Pipe.generate(outflag=outflag): #Pipe is a region generator
            if img is None:
                continue

            mask = argsout.get('mask', None)

            w, h = _ImageInfo.resolution(img)
            test_region, region_img_path, dummy1 = RR.generate(img_path, w, h, 10, mask, outflag=outflag)

            if test_region is None:
                s = 'Failed to generate a test region for %s' % img_path
                _logging.warning(s)
                continue

            yield img, test_region, {'imgpath':img_path, 'region_img_path':region_img_path}




class VOC(_Generator):
    '''Generate images from the Pascal VOC data
    Yields images of a requested category type (train, val or trainval)
    with the bounding boxes from the PASCAL VOC image set.

    category:
        The object category, e.g. cat
    dataset:
        String specifyig the dataset.
        i.e. 'test', 'train', 'val' or 'train_val'
    filters:
        Keyword argument containing a list of filter function from imgpipes.filters, passed as filters=...
    transforms:
        Keyword argument containing lit of ist of imgpipes.transforms functions, passed as transforms=...
        to apply to output image

    Example:
        VOC = generators.VOC('cat', 'train', transforms=T, filters=F)
        for img, file_path, empty_dic in VOC.generate:
        #do some work

    See test/test_generators.py for other examples.
    '''
    def __init__(self, category, *args, dataset='train', **kwargs):
        '''(str, str) -> void
        '''
        self.category = category
        self.dataset = dataset
        self.silent = True
        '''Quite console'''
        super().__init__(*args, **kwargs)


    def generate(self, outflag=_cv2.IMREAD_UNCHANGED, pathonly=False):
        '''(cv2.imread option, bool, bool) -> ndarray|None, str, dict|None
        Yields the images with the bounding boxes and category name of all objects
        in the pascal voc images

        The points and category names are yielded in the dictionary object
        outflag:
            <0 - Loads as is, with alpha channel if present)
            0 - Force grayscale
            >0 - 3 channel color iage (stripping alpha if present
        pathonly:
            only generate image paths, the ndarray will be None

        Yields:
            image, path, {'categories':list of categories, 'rects':CVXY points}
            Example of the dictionary:
                {
                categories:[cat, cat, dog],
                rects:[[0,0],[10,10],[0,10],[10,0]
                        [0,0],[10,10],[0,10],[10,0],
                        [0,0],[10,10],[0,10],[10,0]]
                }

        Note:
            If pathonly is used, the dict will be empty
         '''
        imgs = [x for x in _voc.get_image_url_list(self.category, self.dataset)]
        for fname in imgs:
            if pathonly:
                yield None, fname, None
            else:
                try:
                    img = _cv2.imread(fname, outflag)
                    if not super().isimagevalid(img):
                        continue
                    img = super().generate(img) #delegate to base method to transform and filter (if specified)
                    if isinstance(img, _np.ndarray):
                        cats = []
                        rects = []
                        for cat_name, rect in _voc.regions_generator(fname, self.category):
                            cats.append(cat_name)
                            rects.append(rect)
                        yield img, fname, {'categories':cats, 'rects':rects}
                except Exception as _:
                    s = 'Processing of %s failed.' % fname
                    if not self.silent:
                        print(s)
                    _logging.exception(s)






#region Helper funcs
def _dir_has_vgg(fld):
    if not fld.endswith(VGG_FILE):
        fld = _path.join(_path.normpath(fld), VGG_FILE)
    else:
        fld = _path.normpath(fld)
    return _path.isfile(fld)
#endregion
