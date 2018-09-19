# pylint: disable=C0302, dangerous-default-value, no-member, expression-not-assigned, locally-disabled, not-context-manager, redefined-builtin

'''My file input and output library, e.g. for _csv handling.
Also for general IO to the console'''
from __future__ import print_function as _print_function
from warnings import warn as _warn

import csv as _csv
import glob as _glob
import itertools as _itertools
import os as _os
import time as _time
import shutil as _shutil
import string as _string
import tempfile as _tempfile
from contextlib import contextmanager as _contextmanager
from contextlib import suppress as _suppress
import datetime as _datetime

try:
    import cPickle as _pickle
except BaseException:
    import _pickle

import subprocess as _subprocess
import sys as _sys


from numpy import ndarray as _numpy_ndarray

import fuckit as _fuckit
import funclib.stringslib as _stringslib
from funclib.baselib import get_platform as _get_platform

_NOTEPADPP_PATH = 'C:\\Program Files (x86)\\Notepad++\\notepad++.exe'


class CSVIo(object):
    '''class for reading/writing _csv objects
    can work standalone or as the backbone for CSVMatch'''

    def __init__(self, filepath):
        '''init'''
        self.filepath = filepath
        self.values = []
        self.rows = []

        self.read()

    def read(self, val_funct=lambda val: val):
        '''use val_funct to operate on all the values before as they are read in'''
        with open(self.filepath, 'rU') as f:
            raw_csv = _csv.DictReader(f)
            for row in raw_csv:
                row = {key: val_funct(val) for key, val in row.items()}
                self.rows.append(row)
                self.values += row.values()
            return

    def save(self, filepath=None):
        '''save'''
        if not filepath:
            filepath = self.filepath
        with open(filepath, 'w') as f:
            writer = _csv.DictWriter(f, self.rows[0].keys())
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
            return


class CSVMatch(CSVIo):
    '''CSVMatch class'''

    def row_for_value(self, key, value):
        '''returns a list of matching rows
        key = the column name on the _csv
        value = the value to match in that column

        Returns None if no match
        '''
        if value or not value not in self.values:
            return None

        match = None
        for row in self.rows:
            if row[key] == value:
                if match:
                    raise MultipleMatchError()
                match = row
        return match

    def row_for_object(self, match_function, obj):
        '''
        like row_for_value, but allows for a more complicated match.
        match_function takes three parameters (vals, row, object) and return true/false

        Returns:
            None if no match, else the returns the row
        '''
        for row in self.rows:
            if match_function(row, obj):
                return row
        return None

class MultipleMatchError(RuntimeError):
    '''helper'''
    pass


# region _csv IO
def write_to_eof(filename, thetext):
    '''(_string,_string) ->void
    Write thetext to the end of the file given in filename.
    '''
    with _fuckit:
        with open(filename, 'a+') as fid:
            fid.write(thetext)


def readcsv(filename, cols=1, startrow=0, numericdata=True):
    '''(_string, int, bool, int, bool) -> list
    Reads a _csv file into a list and returns the list
    Set cols to the number of cols in the _csv.

    If you want to skip the first row (eg if you have a header row, set startrow to 1.
    '''
    data = [0] * (cols)
    for i in range(cols):
        data[i] = []
    if _sys.version_info.major == 2:
        with open(filename, 'rb') as csvfile:  # open the file, and iterate over its data
            csvdata = _csv.reader(csvfile)  # tell python that the file is a _csv
            for i in range(0, startrow):  # skip to the startrow
                next(csvdata)
            for row in csvdata:  # iterate over the rows in the _csv
                # Assign the cols of each row to a variable
                for items in range(
                        cols):  # read in the text values as floats in the array
                    if numericdata:
                        data[items].append(float(row[items]))
                    else:
                        data[items].append(row[items])
    elif _sys.version_info.major == 3:
        with open(filename, newline='') as csvfile:  # open the file, and iterate over its data
            csvdata = _csv.reader(csvfile)  # tell python that the file is a _csv
            for i in range(0, startrow):  # skip to the startrow
                next(csvdata)
            for row in csvdata:  # iterate over the rows in the _csv
                # Assign the cols of each row to a variable
                for items in range(
                        cols):  # read in the text values as floats in the array
                    if numericdata:
                        data[items].append(float(row[items]))
                    else:
                        data[items].append(row[items])
    else:
        _sys.stderr.write('You need to use python 2* or 3* \n')
        exit(1)
    return data


def writecsv(filename, datalist, header=[], inner_as_rows=True, append=False, skip_first_row_if_file_exists=False):
    '''(_string, list, list, bool, bool) -> Void
    Reads a _csv file into a list and returns the list
    ---
    inner_as_rows == True
    [[1,a],[2,b]]
    row1:1,2
    row2:a,b
    -
    inner_as_rows == False
    [[1,a],[2,b]]
    row1=1,a
    row2=2,b
    '''
    csvfile = []
    useheader = False
    exists = file_exists(filename)
    if not append:
        exists = False

    try:
        if append:
            csvfile = open(filename, 'a', newline='')
        else:
            csvfile = open(filename, 'w', newline='')
    except FileNotFoundError as e:
        print("Could not create file %s, check the file's folder exists." % filename)
        return
    except Exception as e:
        raise e

    # if user passed a numpy array, convert it
    if isinstance(datalist, _numpy_ndarray):
        datalist = datalist.T
        datalist = datalist.tolist()
    # if there is no data, close the file
    if len(datalist) < 1:
        csvfile.close()
        return
    # check to see if datalist is a single list or list of lists
    is_listoflists = False
    list_len = 0
    num_lists = 0
    if isinstance(datalist[0], (list, tuple)
                  ):  # check the first element in datalist
        is_listoflists = True
        list_len = len(datalist[0])
        num_lists = len(datalist)
    else:
        is_listoflists = False
        list_len = len(datalist)
        num_lists = 1
    # if a list then make sure everything is the same length
    if is_listoflists:
        for list_index in range(1, len(datalist)):
            if len(datalist[list_index]) != list_len:
                _sys.stderr.write(
                    'All lists in datalist must be the same length \n')
                csvfile.close()
                return
    # if header is present, make sure it is the same length as the number of
    # cols
    if header:
        if len(header) != num_lists:
            _sys.stderr.write(
                'Header length did not match the number of columns, ignoring header.\n')
        else:
            useheader = True

    # now that we've checked the inputs, loop and write outputs
    writer = _csv.writer(
        csvfile,
        delimiter=',',
        quotechar='|',
        quoting=_csv.QUOTE_MINIMAL)  # Create writer object
    if useheader:
        writer.writerow(header)
    if inner_as_rows:
        for i, row in enumerate(range(0, list_len)):
            if i == 0 and skip_first_row_if_file_exists and exists:
                pass
            else:
                thisrow = []
                if num_lists > 1:
                    for col in range(0, num_lists):
                        thisrow.append(datalist[col][row])
                else:
                    thisrow.append(datalist[row])
                writer.writerow(thisrow)
    else:
        for i, row in enumerate(datalist):
            if i == 0 and skip_first_row_if_file_exists and exists:
                pass
            else:
                writer.writerow(row)

    # close the _csv file to save
    csvfile.close()
# endregion


# region file system
def temp_folder(subfolder=''):
    '''Returns a folder in the users temporary space.
    subfolder:
        if !== '': create the defined subfolder
        otherwise uses a datetime stamp
    '''
    fld = datetime_stamp() if subfolder == '' else subfolder
    return _os.path.normpath(_os.path.join(_tempfile.gettempdir(), fld))


def datetime_stamp(datetimesep=''):
    '''(str) -> str
    Returns clean date-_time stamp for file names etc
    e.g 01 June 2016 11:23 would be 201606011123
    str is optional seperator between the date and _time
    '''
    fmtstr = '%Y%m%d' + datetimesep + '%H%m%S'
    return _time.strftime(fmtstr)


def exit():
    '''override exit to detect platform'''
    if _get_platform() == 'windows':
        _os.system("pause")
    else:
        _os.system('read -s -n 1 -p "Press any key to continue..."')
    _sys.exit()


def get_platform():
    '''-> str
    returns windows, mac, linux
    '''
    s = _sys.platform.lower()
    if s == "linux" or s == "linux2":
        return 'linux'
    elif s == "darwin":
        return 'mac'
    elif s == "win32" or s == "windows":
        return 'windows'

    return 'linux'


def get_file_count(paths, recurse=False):
    '''(list like|str)->int'''
    cnt = 0

    if isinstance(paths, str):
        paths = [paths]

    for ind, val in enumerate(paths):
        paths[ind] = _os.path.normpath(val)

    if recurse:
        for thedir in paths:
            cnt += sum((len(f) for _, _, f in _os.walk(thedir)))
    else:
        for thedir in paths:
            cnt += len([item for item in _os.listdir(thedir)
                        if _os.path.isfile(_os.path.join(thedir, item))])
    return cnt


def hasext(path, ext):
    '''(str, str|list)->bool
    Does the file have extension ext
    ext can be a list of extensions
    '''
    if isinstance(ext, str):
        return get_file_parts2(path)[2] == ext

    return get_file_parts2(path)[2] in ext


def hasdir(path, fld):
    '''(str, str|list)->bool
    Is the file in folder fld.
    fld can be a list of folders (strings)
    '''
    if isinstance(path, str):
        return get_file_parts2(path)[0] == fld

    return get_file_parts2(path)[0] in fld


def hasfile(path, fname):
    '''(str, str|list)->bool
    Does path contain the filename fname.

    path:
        full path name to a file
    fname:
        the file name

    Example:
    >>>hasfile('c:/tmp/myfile.txt', 'myfile.txt')
    True

    Returns:
        true if fname is the file in path.

    '''
    if isinstance(path, str):
        return get_file_parts2(path)[1] == fname

    return get_file_parts2(path)[1] in fname


def drive_get_uuid(drive='C:', strip=['-'], return_when_unidentified='??'):
    '''get uuid of drive'''
    proc = _os.popen('vol %s' % drive)

    try:
        drive = proc.readlines()[1].split()[-1]
        if not drive:
            drive = return_when_unidentified

        for char in strip:
            drive = drive.replace(char, '')
    except Exception as dummy:
        pass
    finally:
        try:
            proc.close()
        except Exception as dummy:
            pass
        #work

    return drive


def get_file_parts(filepath):
    '''(str)->list[path, filepart, extension]
    Given path to a file, split it into path,
    file part and extension.

    filepath:
        full path to a file.

    Returns:
        The folder, the filename without the extension
        and the extension

    Example:
    >>>get_file_parts('c:/temp/myfile.txt')
    'c:/temp', 'myfile', '.txt'
    '''
    folder, fname = _os.path.split(filepath)
    fname, ext = _os.path.splitext(fname)
    return [folder, fname, ext]


def get_file_parts2(filepath):
    '''(str)->list[path, filepart, extension]
    Given path to a file, split it into path, file part and extension.

    filepath:
        full path to a file.

    Returns:
        The folder, the filename including the extension and the extension

    Example:
    >>>get_file_parts2('c:/temp/myfile.txt')
    'c:/temp', 'myfile.txt', '.txt'
    '''
    folder, fname = _os.path.split(filepath)
    ext = _os.path.splitext(fname)[1]
    return [folder, fname, ext]


def folder_has_files(fld, ext_dotted=[]):
    '''(str, str|list) -> bool
    Does the folder contain files, optionally matching
    extensions. Extensions are dotted.

    Returns false if the folder does not exist.

    fld:
        folder path
    ext_dotted:
        list of extensions to match
    Example:
    >>>folder_has_files('C:/windows')
    True

    >>>folder_has_files('C:/windows', ['.dll'])
    >>>True
    '''
    if isinstance(ext_dotted, str):
        ext_dotted = [ext_dotted]

    for _, _, files in _os.walk(_os.path.normpath(fld)):
        if files and not ext_dotted:
            return True

        for fname in files:
            for ext in ext_dotted:
                if fname.endswith(ext):
                    return True

    return False


def get_available_drives(strip=['-'], return_when_unidentified='??'):
    '''->dictionary
    gets a list of available drives as the key, with uuids as the values
    eg. {'c:':'abcd1234','d:':'12345678'}
    '''
    drives = [
        '%s:' %
        d for d in _string.ascii_uppercase if _os.path.exists(
            '%s:' %
            d)]
    uuids = [drive_get_uuid(drv, strip, return_when_unidentified)
             for drv in drives]
    return dict(zip(drives, uuids))


def get_available_drive_uuids(strip=['-'], return_when_unidentified='??'):
    '''->dictionary
    gets a list of available drives with uuids as the key
    eg. {'c:':'abcd1234','d:':'12345678'}
    '''

    s = _string.ascii_uppercase
    drives = ['%s:' % d for d in s if _os.path.exists('%s:' % d)]
    uuids = [drive_get_uuid(drv, strip, return_when_unidentified)
             for drv in drives]
    return dict(zip(uuids, drives))


def get_drive_from_uuid(uuid, strip=['-']):
    '''str, str iterable, bool->str | None
    given a uuid get the drive letter
    uuid is expected to be lower case

    Returns None if not found
    '''

    for char in strip:
        uuid = uuid.replace(char, '')

    # first val is drive, second is the uuid
    drives = get_available_drive_uuids(strip)
    if uuid in drives:
        return drives[uuid]
    elif uuid.lower() in drives:
        return drives[uuid]

    return None


def folder_generator(paths):
    '''
    (str|iterable)->yield str
    Yields all subfolders
    '''
    if isinstance(paths, str):
        paths = [paths]

    paths = [_os.path.normpath(p) for p in paths]
    for pth in paths:
        for fld, dummy, dummy1 in _os.walk(pth):
            yield fld


def file_list_generator(paths, wildcards):
    '''(iterable, iterable) -> tuple
    Takes a list of paths and wildcards and creates a
    generator which can be used to iterate through
    the generated file list so:
    paths = ('c:/','d:/')     wildcards=('*.ini','*.txt')
    Will generate: c:/*.ini, c:/*.txt, d:/*.ini, d:/*.txt

    ie. Yields wildcards for consumption a _glob.glob.
    '''
    if isinstance(wildcards, str):
        wildcards = [wildcards]

    ww = ['*' + x if x[0] == '.' else x for x in wildcards]

    for vals in (_stringslib.add_right(x[0]) + x[1]
                 for x in _itertools.product(paths, ww)):
        yield _os.path.normpath(vals)


def file_count(paths, wildcards, recurse=False):
    '''(iterable|str, iterable|str, bool) -> int

    Counts files in paths matching wildcards

    paths:
        tuple of list of paths
    wildcards:
        tuple or list of wildcards
    recurse:
        recurse down folders if true
    '''
    cnt = 0
    for _ in file_list_generator1(paths, wildcards, recurse):
        cnt += 1
    return cnt


def file_list_generator1(paths, wildcards, recurse=False):
    '''(str|iterable, str|iterable, bool) -> yields str
    Takes path(s) and wildcard(s), yielding the
    full path to matched files.

    paths:
        Single path or list/tuple of paths
    wildcards:
        Single file extension or list of file extensions.
        Extensions should be dotted, an asterix is appended
        if none exists.
    recurse:
        recurse down folders

    Example syntax:
    >>>for fname in file_list_generator1('C:/temp', '*.txt', recurse=False):
    >>>for fname in file_list_generator1(['C:/temp', 'C:/windows'], ['.bat', '.cmd'], recurse=True):
    '''
    if isinstance(paths, str):
        paths = [paths]

    if isinstance(wildcards, str):
        wildcards = [wildcards]

    wildcards = ['*' + x if x[0] == '.' else x for x in wildcards]

    #for ind, v in enumerate(paths):
       # paths[ind] = _os.path.normpath(v)

    for vals in (_stringslib.add_right(x[0]) + x[1] for x in _itertools.product(paths, wildcards)):
        if recurse:
            for f in file_list_glob_generator(vals, recurse=True):
                yield _os.path.normpath(f)
        else:
            for myfile in _glob.glob(_os.path.normpath(vals)):
                yield _os.path.normpath(myfile)


def file_list_glob_generator(wilded_path, recurse=False):
    '''(str, bool)->yields strings (file paths)
    _glob.glob generator from wildcarded path
    Wilded path would be something like 'c:/*.tmp' or c:/*.*

    Yields actual file names, e.g. c:/temp/a.tmp

    SUPPORTS RECURSION
    '''
    fld, f = get_file_parts2(wilded_path)[0:2]

    if recurse:
        wilded_path = _os.path.normpath(_os.path.join(fld, '**', f))

    for file in _glob.iglob(wilded_path, recursive=recurse):
        yield _os.path.normpath(file)


def files_delete2(filenames):
    '''(list|str) -> void
    Delete file(s) without raising an error

    filenames:
        a string or iterable

    Example:
    >>>files_delete2('C:/myfile.tmp')
    >>>files_delete2(['C:/myfile.tmp', 'C:/otherfile.log'])
    '''
    if isinstance(filenames, str):
        filenames = [filenames]

    for fname in filenames:
        fname = _os.path.normpath(fname)
        if file_exists(fname):
            _os.remove(fname)


def files_delete(folder, delsubdirs=False):
    '''(str)->void
    Delete all files in folder
    '''
    folder = _os.path.normpath(folder)
    if not _os.path.exists(folder):
        return

    for the_file in _os.listdir(folder):
        file_path = _os.path.normpath(_os.path.join(folder, the_file))
        try:
            if _os.path.isfile(file_path):
                _os.unlink(file_path)
            elif _os.path.isdir(file_path):
                if delsubdirs:
                    _shutil.rmtree(file_path)
        except Exception as e:
            print('Could not clear summary file(s). They are probably being used by tensorboard')


def get_temp_fname(suffix='', prefix=''):
    '''get a temp filename'''
    return _tempfile.mktemp(suffix, prefix)


def get_file_name2(fld, ext, length=3):
    '''(str, str, int)-> str
    generate a random filename
    ensuring it does not already exist in
    folder fld.

    Example:
    >>>get_file_name2('C:\temp', '.txt', 4)
    'C:/temp/ABeD.txt'
    '''
    n = 0
    while True:
        s = _os.path.normpath(_os.path.join(fld, '%s%s' % (_stringslib.rndstr(length), ext)))
        if not file_exists(s):
            break
        n += 1
        if n > 100:
            raise StopIteration('Too many iterations creating unique filename')
    return s



def get_file_name(path='', prefix='', ext='.txt'):
    '''(str|None, str, str) -> str
    Returns a filename, based on a datetime stamp

    path:
        path to use, if path='', use CWD,
        if None, then just the filename is returned
    prefix:
        prefix to use
    ext:
        extension
    '''
    if path == '':
        path = _os.getcwd()

    return _os.path.normpath(
        _os.path.join(
            path,
            prefix +
            _stringslib.datetime_stamp() +
            _stringslib.add_left(
                ext,
                _os.path.extsep)))


def folder_open(folder='.'):
    '''(_string) -> void
    opens a windows folder at path folder'''
    if _os.name == 'nt':
        folder = folder.replace('/', '\\')
    with _fuckit:
        _subprocess.check_call(['explorer', folder])


def notepadpp_open_file(filename):
    '''(str) -> void
    opens filename in notepad++

    File name should be in the C:\\dirA\\dirB\\xx.txt format
    '''
    with _fuckit:
        openpth = _NOTEPADPP_PATH + ' ' + '"' + filename + '"'
        _subprocess.Popen(openpth)


def write_to_file(results, prefix='', open_in_npp=True, full_file_path=''):
    '''
    (str|iterable,bool,str) -> str
    Takes result_text and writes it to a file in the cwd.
    Prints out the file name at the end and opens the folder location

    Returns the fully qualified filename written

    Use to quickly right single results set to a file

    If results is a _string then it writes out the _string, otherwise it iterates through
    results writing all elements to the file.

    If full_file_path is defined, saves the file as this, otherwise creates it in CWD with a datetime stamp
    '''
    if full_file_path == '':
        filename = _os.getcwd() + '\\RESULT' + prefix + \
            _stringslib.datetime_stamp() + '.txt'
    else:
        fld, f = full_file_path[0:2]
        create_folder(fld)
        filename = full_file_path

   # n = '\r\n' if _get_platform() == 'windows' else '\n'

    with open(filename, 'w+') as f:
        if isinstance(results, str):
            f.write(results)
        else:
            for s in results:
                f.write(str(s))

    print(results)
    print(filename)
    if open_in_npp:
        if _get_platform() == 'windows':
            notepadpp_open_file(filename)
        else:
            print('Option to open in NPP only available on Windows.')
    return filename


def file_copy(src, dest, rename=False, create_dest=True, dest_is_folder=False):
    '''(str, str, bool, bool, bool) -> str
    Copy a file from src to dest. Optionally
    rename the file if it already exists in dest.

    Can create the dest folder if it doesnt exist even when
    dest is a folder or a full file name.

    src:
       source file path
    dest:
        destination folder or full file path
    rename:
        create a new file if dest exists
    create_dest:
        create the destination folder if it does not exist
    dest_is_folder:
        destination is not a filename but a folder
    Returns:
        the name of the created file

    Example:
    >>>file_copy('c:/temp/myfile.txt', 'c:/temp/newfolder/myfile.txt',
                    create_dest=True,  dest_is_folder=False)
    >>>file_copy('c:/temp/myfile.txt', 'c:/temp/subfolder',
                    create_dest=True, dest_is_folder=True)

    '''
    if dest_is_folder:
        _, fname, _ = get_file_parts2(src)
        if not folder_exists(dest) and create_dest:
            _os.mkdir(dest)
        dest = _os.path.join(dest, fname)
    else:
        pth, fname, ext = get_file_parts(dest)
        if not folder_exists(pth) and create_dest:
            _os.mkdir(pth)

    cnt = 0
    if rename and file_exists(dest):
        pth, fname, ext = get_file_parts(dest)
        dest =  _os.path.join(pth, '%s%s%s' % (fname, _stringslib.rndstr(4), ext))
        while file_exists(dest):
            cnt +=1
            dest = _os.path.join(pth, '%s%s%s' % (fname, _stringslib.rndstr(4), ext))
            if cnt > 1000: #safety
                break

    _shutil.copy2(src, dest)
    return dest


def file_create(file_name, s=''):
    '''(str, str) -> void
    Creates file  and write s to it
    if it doesnt exist
    '''
    if not _os.path.isfile(file_name):

        write_to_eof(file_name, s)


def fixp(pth):
    '''(str)->str
    basically path.normpath
    '''
    return _os.path.normpath(pth)


def file_exists(file_name):
    '''(str) -> bool
    Returns true if file exists
    '''
    if isinstance(file_name, str):
        return _os.path.isfile(fixp(file_name))

    return False


def folder_exists(folder_name):
    '''check if folder exists'''
    if isinstance(folder_name, str):
        return _os.path.exists(fixp(folder_name))

    return False


def create_folder(folder_name):
    '''(str) -> void
    creates a folder
    '''
    if not _os.path.exists(folder_name):
        _os.makedirs(folder_name)


def pickleit(full_file_name, obj):
    '''(str) -> void
    Takes full_file path and pickles (dumps) obj to the file system.
    Does a normpath on full_file_name
    '''
    with open(_os.path.normpath(full_file_name), 'wb') as myfile:
        _pickle.dump(obj, myfile)


def unpickle(path):
    '''(str) -> unpickled stuff
    attempts to load a pickled object named path
    Returns None if file doesnt exist
    '''
    if file_exists(path):
        with open(path, 'rb') as myfile:
            return _pickle.load(myfile)
    else:
        return None
# endregion


# region console stuff
def input_int(prompt='Input number', default=0):
    '''get console input from user and force to int'''
    try:
        inp = input
    except NameError:
        pass
    return int(_stringslib.read_number(inp(prompt), default))


def print_progress(
        iteration,
        total,
        prefix='',
        suffix='',
        decimals=2,
        bar_length=30):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix _string (Str)
        suffix      - Optional  : suffix _string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        bar_length   - Optional  : character length of progbar (Int)
    """
    if total == 0:
        _warn('Total iterations was set to zero.')
        return

    filled_length =  int(round(bar_length * iteration / float(total))) if total > 0 else 0
    if iteration / float(total) > 1:
        total = iteration
    percents = round(100.00 * (iteration / float(total)), decimals)
    if bar_length > 0:
        progbar = '#' * filled_length + '-' * (bar_length - filled_length)
    else:
        progbar = ''

    _sys.stdout.write(
        '%s [%s] %s%s %s\r' %
        (prefix, progbar, percents, '%', suffix)), _sys.stdout.flush()
    if iteration == total:
        print("\n")

#In the consider using tqdm
class PrintProgressFlash(object):
    '''class to print a progress flasher
    to console

    args
        ticks:  max size of chars before reset,
                set to None to print fixed length flasher
        msg:
                print msg to console
    '''

    def __init__(self, ticks=None, msg='\n'):
        '''init'''
        self.ticks = ticks
        print(msg)

    def update(self):
        '''update state'''
        secs = _datetime.datetime.timetuple(_datetime.datetime.now()).tm_sec

        if self.ticks is None:
            if secs % 3 == 0:
                s = '////'
            elif secs % 2 == 0:
                s = '||||'
            else:
                s = '\\\\\\\\'
        else:
            n = int(self.ticks*(secs/60))
            s = '#' * n  + ' ' * (self.ticks - n) #print spaces at end



        _sys.stdout.write('%s\r' % s)
        _sys.stdout.flush()

#In the future consider using tqdm
class PrintProgress(object):
    '''Class for dos progress bar. Implement as global for module level progress

    Example:
        from funclib.iolib import PrintProgress as PP
        pp = PP(len(_glob(img_path)))
        pp.iteration = 1
        pp.increment
    '''

    def __init__(self, maximum=0, bar_length=30, init_msg='\n'):
        print(init_msg)
        self.max = maximum
        self.bar_length = bar_length
        self.iteration = 1

    def increment(self, step=1, suffix=''):
        '''(int) -> void
        advance the counter step ticks.
        1 will usually make sense!'''
        print_progress(
            self.iteration, self.max, prefix='%i of %i' %
            (self.iteration, self.max), bar_length=self.bar_length, suffix=suffix)
        self.iteration += step

    def reset(self, max=None):
        '''reset the counter. set max
        if need to change total expected
        iterations.
        '''
        if max:
            self.max = max
        self.iteration = 1


# endregion


def wait_key(msg=''):
    ''' (str) -> str
    Wait for a key press on the console and returns it.
    msg:
        prints msg if not empty
    '''
    result = None
    if msg:
        print(msg)

    if _os.name == 'nt':
        import msvcrt
        result = msvcrt.getch()
    else:
        import termios as _termios
        fd = _sys.stdin.fileno()

        oldterm = _termios.tcgetattr(fd)
        newattr = _termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~_termios.ICANON & ~_termios.ECHO
        _termios.tcsetattr(fd, _termios.TCSANOW, newattr)

        try:
            result = _sys.stdin.read(1)
        except IOError:
            pass
        finally:
            _termios.tcsetattr(fd, _termios.TCSAFLUSH, oldterm)

    if isinstance(result, bytes):
        return result.decode()

    return result


@_contextmanager
def quite(stdout=True, stderr=True):
    '''(bool, bool) -> void
    Stop messages and errors being sent to the console
    '''
    with open(_os.devnull, "w") as devnull:
        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        if stdout:
            _sys.stdout = devnull

        if stderr:
            _sys.stderr = devnull
        try:
            yield
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr


def time_pretty(seconds):
    '''(float) -> str
    Return a prettified time interval
    for printing
    '''
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd %dh %dm %ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh %dm %ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm %ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)