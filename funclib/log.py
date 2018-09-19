# pylint: disable=C0103, too-few-public-methods, locally-disabled, no-self-use
'''Global logger
Create in __init__ (with error handling).
'''

import logging as _logging
import logging.handlers as _handlers
import tempfile as _tempfile
from os.path import normpath as _normpath
import os as _os


class RootLogger(object):
    '''global logger

    Instantiate in a packages __init__ for global use
    from the __init__log snippet
    '''
    _LOGGER_NAME = 'root'
    USER_TEMP_FOLDER = _normpath(_tempfile.gettempdir())

    def __init__(self, logpath, sizeKB=50, nrbaks=3):
        '''(str)
        full log file path
        e.g. C:\temp\app.log
        '''
        self._logpath = logpath
        self._sizeKB = sizeKB
        self._nrbaks = nrbaks
        self._load()

    def __str__(self):
        info = _os.stat(self._logpath)
        kb = lambda x: '{:n}'.format(x/1024) + ' Kb'
        s = ('Logging to file: %s\n'
            'Max Size: %s\n'
            'Current Size: %s\n'
            'Remaining: %s' % (self._logpath, kb(self._sizeKB), kb(info.st_size), kb(self._sizeKB - info.st_size)))
        return s


    def _load(self):
        # logger settings
        log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
        log_filemode = "w" # w: overwrite; a: append

        _logging.basicConfig(filename=self._logpath, format=log_format, filemode=log_filemode, level=_logging.DEBUG)
        rotate_file = _handlers.RotatingFileHandler(self._logpath, maxBytes=self._sizeKB, backupCount=self._nrbaks)
        logger = _logging.getLogger(self._LOGGER_NAME)
        logger.addHandler(rotate_file)

        #print log messages to console
        consoleHandler = _logging.StreamHandler()
        logFormatter = _logging.Formatter(log_format)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
        logger.propagate
        self.logger = logger
