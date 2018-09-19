# pylint: disable=C0103, too-few-public-methods, locally-disabled,
# no-self-use, unused-argument
'''handles getting additional region data generated from vgg jason files
http://www.robots.ox.ac.uk/~vgg/software/via/

Note that this assumes that the VGG file is in the same folder
as the images for the key fix hack to work

Coordinate system for VGG is:
x = Columns, with origin at left
y = Rows, with origin at top

So: 50x50 image. Top Left x=0,y=0: Bottom Right x=50, y=50


Example:
Iterate over every image
and region specified in the vgg JSON file.

for img in vgg.
img = vgg.Image(
    'C:/Users/Graham Monkman/OneDrive/Documents/PHD/images/bass' \
    '/angler/10150756_851703354845619_1559938228217274444_n.jpg'
    )
assert isinstance(img, vgg.Image)
for subject in img.subjects_generator('bass'):
    assert isinstance(subject, vgg.Subject)
    for region in subject.regions_generator():
        assert isinstance(region, vgg.Region)
        print(region.species, region.part, region.shape)
'''

import os.path as _path
import json
import logging
from shutil import copy as _copy
from math import pi as _pi

from funclib.stringslib import datetime_stamp as _datetime_stamp
from funclib.baselib import dictp as _dictp

from funclib.iolib import get_file_parts as _get_file_parts
from funclib.iolib import get_file_parts2 as _get_file_parts2

from funclib.iolib import print_progress as _print_progress
from funclib.iolib import file_exists as _file_exists
from funclib.baselib import list_not as _list_not
from funclib.baselib import list_and as _list_and
from funclib.baselib import dic_match as _dic_match
from funclib.baselib import eDictMatch as _eDictMatch
import opencvlib.roi as _roi
from opencvlib.info import ImageInfo as _ImageInfo

SILENT = True
_JSON_FILE_NAME = ''
JSON_FILE = []
_LOG_FILE_NAME = 'vgg.py.' + _datetime_stamp() + '.log'

VALID_SHAPES_ALL = ['', 'polygon', 'rect', 'circle', 'ellipse', 'point']
VALID_SHAPES_2D = ['polygon', 'rect', 'circle', 'ellipse']

VALID_PARTS = ['', 'antenna', 'abdomen', 'body', 'cephalothorax',
               'claw', 'exclude', 'head', 'perelopods-legs',
               'telson-tail', 'thorax', 'whole',
               'anal fin', 'caudal fin',
               'dorsal soft fin', 'dorsal spiny fin',
               'pectoral fin', 'pelvic fin',
               'caudal-peduncle', #the join between the body and the caudal fin
               'body', #tight to body, excluding fins
               'body-caudal' #tight to body, but includes caudal
               ]



VALID_SPECIES = ['bass',
                 'cod',
                 'crab-edible',
                 'dab',
                 'dogfish',
                 'flatfish',
                 'flounder',
                 'gurnard grey',
                 'lobster',
                 'mackerel',
                 'plaice'
                 ]


def _prints(s):
    if not SILENT:
        print(s)


logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=_LOG_FILE_NAME,
                    filemode='w',
                    level=logging.DEBUG)


_prints('Logging to %s' % _LOG_FILE_NAME)


def _valid_parts_in_list(parts, silent=False):
    '''(iterable)->list
    Return valid parts in parts against the master list
    '''
    for r in _list_not(parts, VALID_PARTS):
        if not silent:
            print('Part %s invalid' % r)
    return _list_and(parts, VALID_PARTS)


def _valid_species_in_list(species, silent=False):
    '''(iterable)->list
    check if a part is in the valid parts list
    '''
    for r in _list_not(species, VALID_SPECIES):
        if not silent:
            print('Species %s invalid' % r)
    return _list_and(species, VALID_SPECIES)


def fix_keys(backup=True, show_progress=False, del_if_no_file=False):
    '''(bool, bool, bool) -> void
    Loop through json file and correct file sizes
    so the keys are correct. Keys have to be correct
    or else they won't be loaded when editing in VGG.

    backup:
        make a backup of the vgg file
    show_progress:
        print progress status to the console
    del_if_no_file:
        If the file found in the VGG file does not exist
        then delete the corresponding key
    '''
    cnt = 0
    _prints('\n\nFixing keys...\n')
    dkeys = []
    for key in JSON_FILE:
        # relies on VGG JSON file being in same dir as files
        filepath = _get_file_parts(_JSON_FILE_NAME)[0]
        filename = JSON_FILE[key]['filename']
        fullname = _path.join(filepath, filename)
        #only fix if we find it!
        if _path.isfile(fullname):
            size_in_bytes = _path.getsize(fullname)
            newkey = filename + str(size_in_bytes)
            JSON_FILE[newkey] = JSON_FILE.pop(key)
        else:
            dkeys.append(key)

        if not SILENT or show_progress:
            cnt += 1
            s = '%s of %s' % (cnt, len(JSON_FILE))
            _print_progress(cnt, len(JSON_FILE), s, bar_length=30)

    if del_if_no_file:
        for key in dkeys:
            del JSON_FILE[key]

    save_json(backup)
    _prints('\n\nKey fixing complete.\n')


def imagesGenerator(skip_imghdr_check=False, file_attr_match=None, json_file=None, error_on_missing_image=False):
    '''(bool, dict)
    Generate VGG.Image class objects
    for every image in the vgg file

    skip_imghdr_check:
        Perform simple file exists check instead of using
        the imghdrlibrary to determine if the file is a
        valid image.
    file_attr_match:
        Only yield images which match or partially match
        this dictionary, e.g. file_attr_match={'istrain':1}.

        If file_attrs_match is provided, then images with
        no file attributes will NOT be yielded.

        NOTE: All dict values read will be a string
    json_file:
        Open this json file, saves calling vgg.load_json first
    error_on_missing_image:
        Raise a FileNotFoundError if the JSON file references an
        image which cannot be found.

    Comments:
        Use the skip_imghdr_check if valid
        image files are being skipped.

    Yields:
        VGG.Image class intances
    '''
    if isinstance(json_file, str):
        load_json(json_file)

    for img in JSON_FILE:
        pth = _get_file_parts2(_JSON_FILE_NAME)[0]
        img_pth = _path.join(pth, JSON_FILE[img]['filename'])

        if error_on_missing_image and not _file_exists(img_pth):
            raise FileNotFoundError('Image file %s not found' % img_pth)

        doit = False
        if isinstance(file_attr_match, dict):
            file_attrs = JSON_FILE[img].get('file_attributes', '')
            if file_attrs and file_attrs != '':
                m = _dic_match(file_attr_match, file_attrs)
                doit = (m == _eDictMatch.Exact or m == _eDictMatch.Subset)
            else:
                doit = False #be explicit - if we have attrs, but no attrs defined we dont want it
        else:
            doit = True
        i = None
        if doit:
            if skip_imghdr_check:
                if _file_exists(img_pth):
                    i = Image(img_pth)
            else:
                if _ImageInfo.is_image(img_pth):
                    i = Image(img_pth)

        if isinstance(i, Image):
            yield i



def roiGenerator(json_file=None, skip_imghdr_check=False, shape_type=None, file_attr_match=None, region_attr_match=None):
    '''str, bool,  tuple|list|str|None, dict|None, dict|None -> str, 2-tuple, Class:Region
    Simplified wrapper for generating regions.

    Yields image filename, the image resolution (w, h) and the region class
    representation of the roi.

    json_file:
        name of the vgg json file, if passed, this will load the pass file
        otherwise the currently loaded file will be searched
    skip_imghdr_check:
        Do a simple file exists test, rather than advanced check of image file
        validity, which can have false negatives
    shape_type:
        The type of shape, 'rect', 'circle', 'point', 'ellipse'.
        Also accepts a list or tuple. If None, then all shapes yielded.
    file_attr_match:
        Only yield images which match or partially match
        this dictionary, e.g. file_attr_match={'istrain':1}.

        If file_attrs_match is provided, then images with
        no file attributes will NOT be yielded.

        NOTE: All dict values read will be a string
    region_attr_match:
        A dictionary which is matched against the region attributes
        of the shape. If not a dictionary, then all shapes will
        be yielded.
        e.g. region_attr_match={'part':'head'}
    '''
    for Img in imagesGenerator(skip_imghdr_check=skip_imghdr_check, file_attr_match=file_attr_match, json_file=json_file):
        assert isinstance(Img, Image)
        for Reg in Img.roi_generator(shape_type=shape_type, region_attr_match=region_attr_match):
            assert isinstance(Reg, Region)
            yield Img.filepath, Img.resolution, Reg


class Image(object):
    '''Load and iterate VGG configured image regions
    based on the actual image file name, size and path

    filepath:
        full path to file, e.g. c:/tmp/img.jpg
    fileext:
        file extension, e.g. .jpg
    filename:
        just the file name, e.g. img.jpg
    filefolder:
        the folder which contains the image file, e.g. c:/tmp
    points:
        all points not associated with a region
    shape_count:
        number of shapes associated with attributes
    subject_count:
        number of subject, i.e. a detection object like a face or fish
    '''

    def __init__(self, filepath=''):
        '''(str)
        optionally provide a filepath - ie a path including the file name
        '''
        self.filepath = filepath
        self._size_in_bytes = 0
        self._key = ''
        self.filename = ''
        self.fileext = ''
        self.filefolder = ''
        #self.key_ignores_filesize = key_ignores_filesize
        self._load_image(filepath)


    def _get_key(self):
        '''->[str | None]
        Generate unique key for image file
        Returns None if image file does not exist
        or an error occurs
        '''
        if _path.isfile(self.filepath):
            try:
                self._size_in_bytes = _path.getsize(self.filepath)
                # return self.filename + str(self.size_in_bytes)
                return self.filename  # Not using exact key as file sizes can change when image tags change
            except Exception:
                log = 'Failed to generate key for file %s' % self.filepath
                logging.warning(log)
                return None
        else:
            log = 'File not found %s' % self.filepath
            logging.warning(log)
            return None


    def _load_image(self, filepath):
        '''(str)->void
        set key and instance variables relating to the file path

        filepath:
            full path to the image
        '''
        self.filepath = filepath
        if filepath != '':
            self.filefolder, self.filename, self.fileext = _get_file_parts(
                _path.abspath(_path.normpath(self.filepath)))
            self.filename = self.filename + self.fileext
            self._key = self._get_key()


    @property
    def resolution(self):
        '''() ->  2-tuple
        Gets the resolution as (w, h)

        If it fails returns (None, None)
        '''
        h = None
        w = None
        try:
            w, h = _ImageInfo.getsize(self.filepath)
        except Exception as _:
            pass
        return (w, h)


    def subjects_generator(self, species):
        '''(str)->Class:Subject
        Subjects generator for the specied species

        Yields
            Class:Subjects
        '''
        subjectids = []

        if species not in VALID_SPECIES:
            raise ValueError('Invalid species ' + species)

        d = _dictp(JSON_FILE)
        regions = d[self._key]['regions']

        if regions:
            assert isinstance(regions, dict)
            for region in regions.values():
                if region.get('region_attributes').get('species', '').casefold() == species.casefold():
                    subjectid = region.get(
                        'region_attributes').get('subjectid')
                    if subjectid not in subjectids:
                        subjectids.append(subjectid)
                        sbj = Subject(self._key, subjectid)
                        yield sbj
        else:
            s = 'No VGG regions defined for image %s' % self.filepath

            if not SILENT:
                print(s)
            logging.warning(s)


    def roi_generator(self, shape_type=None, region_attr_match=None):
        '''(tuple|list|str|None, dic) -> Class:Region
        Class region represents a shape and contains the
        shape attributes, e.g. points and shape type. It
        has no idea of a subject, where a subject represents
        an object of interest, which may have multiple
        associated ROIs.

        This is a more general version of the subjects generator
        which is currently specific to defined fish species.

        shape_type:
            The type of shape, 'rect', 'circle', 'point', 'ellipse'.
            Also accepts a list or tuple. If None, then all shapes yielded.
        region_attr_match:
            A dictionary which is matched against the region attributes
            of the shape. If not a dictionary, then all shapes will
            be yielded.

        Yields:
            Class:Region
        '''
        if isinstance(shape_type, str):
            shape_type = [shape_type]

        d = _dictp(JSON_FILE)
        regions = d[self._key]['regions']
        if regions:
            assert isinstance(regions, dict)
            for i, region in enumerate(regions.values()):
                region_json_key = list(regions.keys())[i]
                region_attrs = region.get('region_attributes', None)
                shape_attrs = region.get('shape_attributes', None)

                assert isinstance(region_attrs, dict)
                assert isinstance(shape_attrs, dict)

                if shape_attrs is None:
                    continue

                if not shape_attrs.get('name') in shape_type:
                    continue

                reg = None
                regionid = region_attrs.get('regionid')
                if not isinstance(region_attr_match, dict):
                    reg = _load_region(shape_attrs, region_attrs, self._key, regionid, region_json_key)
                else: #we have asked for a filter
                    m = _dic_match(region_attr_match, region_attrs)
                    if m == _eDictMatch.Exact or m == _eDictMatch.Subset:
                        reg = _load_region(shape_attrs, region_attrs, self._key, regionid, region_json_key)

                if isinstance(reg, Region):
                    yield reg



    @property
    def image_points(self):
        '''() -> list

        Return all points in the image not associated with a subject,
        i.e. points with no shape attributes defined.


        Returns:
            List of points in CVXY format, i.e. [[1,2], [50,30], ....]
        '''
        #Dev Note, this could also return a dictionary of the data
        #associated with the point

        d = _dictp(JSON_FILE)
        regions = d[self._key]['regions']

        if not regions:
            return None

        assert isinstance(regions, dict)
        pts = []
        for region in regions.values():
            if region.get('shape_attributes').get('name', '').casefold() == 'point' and region.get('region_attributes') == {}:
                shape_attr = region.get('shape_attributes')
                if not shape_attr:
                    return None

                try:
                    cx = int(shape_attr.get('cx', ''))
                    cy = int(shape_attr.get('cy', ''))
                    pts.append([cx, cy])
                except Exception:
                    pass

        if pts:
            return pts
        return None


    @property
    def subject_count(self, species=''):
        '''str->int
        Returns number of valid regions

        If species is set, checks for that species only
        '''
        cnt = 0

        for spp in VALID_SPECIES:
            if species.casefold() == spp.casefold() or species == '':
                for dummy in self.subjects_generator(spp):
                    cnt += 1
        return cnt


    @property
    def shape_count(self):
        '''->int
        Returns number of shapes
        '''
        d = _dictp(JSON_FILE)
        regions = d[self._key]['regions']
        cnt = 0
        if regions:
            assert isinstance(regions, dict)
            for region in regions.values():
                if region.get('shape_attributes'):
                    cnt += 1
        return cnt


class Subject(object):
    '''really a fish object, has many regions
    Should not be accessed directly.
    Iterate through the Images class subjects_generator.

    If no subjectid is used to initialise the class, then
    the iterator will ignore subject assignmensts, ie all
    regions are treated as belonging to the same subject (fish)
    '''

    def __init__(self, key, subjectid=None):
        '''(str, str)
        Key is the unique key for the image,
        subjectid is set as an integer to uniquely identify a subject
        '''
        self.key = key
        self.subjectid = subjectid
        self.region_ids = set([])
        self.set_regions()

    def set_regions(self):
        '''
        Checks all regions defined on the image, regions which
        are defined on the same Object/Subject (by subjectid set in VGG) have their region key
        saved to region_ids for access later
        '''

        d = _dictp(JSON_FILE)
        regions = d.getp(self.key).get('regions')

        for key, region in regions.items():
            # if called with no subject id, just yield all the regions in
            # regions generator,
            # this ignores subject and assumes there is only one fish.
            # This is used write_region_attributes
            assert isinstance(region, dict)
            if self.subjectid is None:
                self.region_ids.add(key) #dont care about a subject, so get all regions without checking if they match a subject
            else:
                rattr = region.get('region_attributes')
                if rattr is None:
                    s = 'No region_attributes for file with key %s' % self.key
                    logging.warning(s)
                    if not SILENT:
                        print(s)
                else:
                    subj = rattr.get('subjectid')
                    if  subj is None:
                        s = 'No subjectid set in region_attributes for file with key %s' % self.key
                        logging.warning(s)
                        if not SILENT:
                            print(s)
                    else:
                        if subj == self.subjectid:
                            self.region_ids.add(key)


    def regions_generator(self, part='', shape=''):
        '''(str, str)->Yields Region classes
        Yields regions for the given subject in the photo
        (identifed by its subjectid) which is unique within image only.

        If part == '', yields all parts associated with the object/subject
        Otherwise only yields the part specified (e.g., head or whole)

        If shape is specified, only yields regions of the corresponding shape

        If no region attributes assumes region is the whole object ('whole')
        '''
        if part not in VALID_PARTS:
            raise ValueError('Invalid image region type (part) ' + part)

        if shape not in VALID_SHAPES_ALL:
            raise ValueError('Invalid shape ' + part)

        d = _dictp(JSON_FILE)
        regions = d.getp(self.key).get('regions')

        # region_ids is the ids of all regions which share a common
        # region_attributes['subjectid']
        for region_key in self.region_ids:
            shape_attr = regions.get(region_key).get('shape_attributes')
            if shape_attr is None:
                s = 'No shape_attributes for file with key %s' % self.key
                logging.warning(s)
                if not SILENT:
                    print(s)
                continue

            region_attr = regions.get(region_key).get('region_attributes')
            if region_attr is None:
                s = 'No region_attributes for file with key %s' % self.key
                logging.warning(s)
                if not SILENT:
                    print(s)
                continue

            if shape_attr:
                reg = _load_region(shape_attr, region_attr, self.key, region_key)
                if part == '' or part.casefold() == region_attr['part'].casefold(): #only get parts we ask for - this is a custimisation
                    if shape == '' or shape.casefold() == shape_attr.get('name'): #only get shapes we ask for
                        yield reg


class Region(object):
    '''Object representing a a single shape marked on a subject,
    like a head or the whole.

    Note opencv points have origin in top left and are (x,y) ie col,row (width,height). Not the matrix standard.
    '''

    def __init__(self, **kwargs):
        '''supported kwargs
        name= [circle | polygon | rect]
        object_part = [head, whole, ....]

        circle: x,y,r
        rect: x,y,w,h
        polygon: [all_points_x], [all_points_y]  [10,20,50], [30,50, 100]

        values set to None if not read

        Coordinate system for VGG (which is the same as CVXY in opencv) is:
        x = Columns, with origin at left
        y = Rows, with origin at top

        So: 50x50 image. Top Left x=0,y=0: Bottom Right x=50, y=50

        bounding_rectangle property is the x,y,w,h representation of a rectangle

        Note that the entire region attributes dictionary is also suffed into Region.region_attr
        '''
        self.region_json_key = kwargs.get('region_json_key', None) #this is the key for the region as read directly from the JSON
        self.region_attr = kwargs.get('region_attr', None) #store the entire region attributes anyway, just incase we need them for future use
        self.image_key = kwargs.get('image_key', None)
        self.has_attrs = kwargs.get('has_attrs', False)
        # should never be None (but can be an empty dict), so error if not
        # present
        self.region_key = kwargs.get('region_key', None)
        self.shape = kwargs.get('shape')
        self.species = kwargs.get('species', None)
        self.part = kwargs.get('part', None)
        self.subjectid = kwargs.get('subjectid', None)
        self.area = 0
        if self.shape == 'rect':
            self.x = kwargs.get('x')
            self.y = kwargs.get('y')
        else:  # points, ellipses, circles
            self.x = kwargs.get('cx')
            self.y = kwargs.get('cy')

        # ellipse only
        self.rx = kwargs.get('rx')
        self.ry = kwargs.get('ry')

        # circles only
        self.r = kwargs.get('r')

        # rect
        self.w = kwargs.get('w')
        self.h = kwargs.get('h')

        self.w = kwargs.get('width') if self.w is None else self.w
        self.h = kwargs.get('height') if self.h is None else self.h
        # polygon
        self.all_points_x = kwargs.get('all_points_x', None)
        self.all_points_y = kwargs.get('all_points_y', None)

        if self.shape == 'polygon':
            self.all_points = list(zip(self.all_points_x, self.all_points_y))
            self.area = _roi.poly_area(pts=self.all_points)
            self.bounding_rectangle_as_points = _roi.bounding_rect_of_poly(
                self.all_points, as_points=True)
            self.bounding_rectangle_xywh = _roi.bounding_rect_of_poly(self.all_points, as_points=False) # x,y,w,h
        elif self.shape == 'point':
            self.all_points = [(self.x, self.y)]
            self.bounding_rectangle_as_points = None
        elif self.shape == 'rect':
            self.all_points = _roi.rect_as_points(self.y, self.x, self.w, self.h)
            self.area = _roi.poly_area(pts=self.all_points)
            self.all_points_x, self.all_points_y = zip(*self.all_points)
        elif self.shape == 'circle':
            self.area = _pi * self.r ** 2
            self.bounding_rectangle_as_points = _roi.bounding_rect_of_ellipse(
                (self.x, self.y), self.r, self.r)  # circle is just an ellipse
            self.bounding_rectangle_xywh = [
                self.rx - self.r, self.ry - self.r, self.r * 2, self.r * 2]
        elif self.shape == 'ellipse':
            self.area = _pi * self.rx * self.ry
            self.bounding_rectangle_as_points = _roi.bounding_rect_of_ellipse(
                (self.x, self.y), self.rx, self.ry)
            self.bounding_rectangle_xywh = [
                self.rx - self.rx, self.ry - self.ry, self.rx * 2, self.ry * 2]

    def write(self):
        '''->void
        write the region to the in memory json file _JSON_FILE
        '''
        JSON_FILE[self.image_key]['regions'][self.region_key]['region_attributes']['subjectid'] = str(
            self.subjectid)
        JSON_FILE[self.image_key]['regions'][self.region_key]['region_attributes']['species'] = self.species
        JSON_FILE[self.image_key]['regions'][self.region_key]['region_attributes']['part'] = self.part


def _load_region(shape_attr, region_attr=None, image_key=None, region_key=None, region_json_key=None):
    '''(dict, str|None, str|None, dict|None)-> Class:Region
    Make a region object from the dictionary
    representations read from the VGG file.

    shape_attr:
        shape_attributes dictionary from VGG
    region_attr:
        region_attributes dictionary from VGG

    region_json_key:
        The dictionary key for the region, as stored in the
        json file. The "1" in >>> "regions": { "1": { ....
        "shape

    Returns:
        Class instance of Region.

    Notes:
        Entire region attributes (if present)
        are stored in the region class as region_attr
    '''
    if not isinstance(shape_attr, dict):
        return None

    reg = Region(region_json_key=region_json_key, #the key for the region as stored in the json
                region_attr=region_attr, #store all the region attributes incase we need them
                part=region_attr.get('part'),
                image_key=image_key,  # no get, error if doesnt exist
                has_attrs=True if region_attr else False,
                region_key=region_key,  # no get, error if doesnt exist
                species=region_attr.get('species'),
                subjectid=region_attr.get('subjectid'),
                shape=shape_attr.get('name'),
                x=shape_attr.get('x'),
                y=shape_attr.get('y'),
                r=shape_attr.get('r'),
                w=shape_attr.get('width'),
                h=shape_attr.get('height'),
                cx=shape_attr.get('cx'),
                cy=shape_attr.get('cy'),
                rx=shape_attr.get('rx'),
                ry=shape_attr.get('ry'),
                all_points_x=shape_attr.get('all_points_x'),
                all_points_y=shape_attr.get('all_points_y'))
    return reg


# region Save and Load the file
def load_json(vgg_file, fixkeys=True, backup=True):
    '''(str)->void
    Load the VGG JSON file into the module level variable _JSON_FILE

    vgg_file:
        path to the vgg json file
    '''
    pth = _path.normpath(vgg_file)
    global JSON_FILE
    global _JSON_FILE_NAME
    with open(pth) as data_file:
        JSON_FILE = json.load(data_file)
    _JSON_FILE_NAME = pth
    if fixkeys:
        fix_keys(backup)


def save_json(backup=True):
    '''->void
    called after we have fixed the keys in the JSON file
    to reflect any changes in image filesize
    caused by changing tags

    If backup then the file is backed up before writing out
    the in memory JSON file
    '''
    filepath, filename, ext = _get_file_parts(_JSON_FILE_NAME)
    if backup:
        bk = _path.join(filepath, filename + _datetime_stamp() + ext + '.bak')
        _copy(_JSON_FILE_NAME, bk)
        s = '\nCreated backup of VGG file %s' % bk
        _prints(s)
        logging.info(s)
    with open(_JSON_FILE_NAME, 'w') as outfile:
        json.dump(JSON_FILE, outfile)
    s = '\nSaved VGG JSON file %s' % _JSON_FILE_NAME
    _prints(s)
    logging.info(s)
# endregion
