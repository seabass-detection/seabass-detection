# pylint: disable=C0302, no-member, expression-not-assigned,
# not-context-manager
''' helper for interacting with application ini files'''
import configparser as _cp
import os as _os
import ast as _ast
from enum import Enum as _Enum


import funclib.iolib as _iolib


class eReadAs(_Enum):
    '''enum'''
    ersDict = 1
    ersList = 2
    ersStr = 3
    ersTuple = 4


class ConfigFile(object):
    '''handles ini file defaults for common sections and values
    like data paths etc
    '''

    def __init__(self, ini_file):
        self.ini_file_path, self.ini_file_name = _os.path.split(
            _os.path.abspath(ini_file))
        self.ini_file = ini_file
        if not str(ini_file).endswith('ini') and not str(ini_file).endswith('cfg'):
            raise ValueError('Expected ini file to have extension ini or cfg')

        if not _os.path.isfile(ini_file):
            _iolib.file_create(ini_file)
        self._config = _cp.ConfigParser()
        self._config.read(ini_file)

    def __str__(self):
        assert isinstance(self._config, _cp.ConfigParser)
        return str(self._config.options)


    def tryread(self, section, option, force_create=False, value_on_create='', asType=eReadAs.ersStr, error_on_read_fail=False):
        '''(str, str, bool, str|dict, Enum:eReadAs, bool) -> str
        Returns the value read, which will default to value_on_create if no section or option is found.
        Saves to disk if new option created.

        section:
            The section [DEFAULT]
        option:
            The key of an attribute
        force_create:
            Create the section and option with value value_on_create
        asType:
            The type to try to load the value as, so we can force
            reading a value in the config file as a dictionary for example
        error_on_read_fail:
            Raise KeyError if entry not read
        '''
        assert isinstance(self._config, _cp.ConfigParser)
        if self._config.has_section(section): #have the section eg [CONFIG]
            if self._config.has_option(section, option): #has entry, eg mysetting:3
                s = self._config.get(section, option)
                if asType != eReadAs.ersStr:
                    d = _ast.literal_eval(s)
                    return d
                return s
            else:
                if force_create:
                    if isinstance(value_on_create, dict):
                        self._config.set(section, option, str(value_on_create))
                    else:
                        value_on_create = value_on_create
                    self.save()
                    return value_on_create
                else:
                    raise KeyError('Option %s not found for section %s in inifile %s' % (option, section, self.ini_file))
        else:
            if force_create:
                self._config.add_section(section)
                if isinstance(value_on_create, dict):
                    self._config.set(section, option, str(value_on_create))
                else:
                    self._config.set(section, option, value_on_create)
                self.save()
                return value_on_create
            else:
                raise KeyError('Section %s not found in %s' % (section, self.ini_file))


    def trywrite(self, section, option, value):
        '''(str,str,str) ->void
        tries to write out a value to ini, creating new section if
        the section doesnt already exist
        Saves to disk once done
        '''
        if not self._config.has_section(section):
            self._config.add_section(section)
        self._config.set(section, option, str(value))


    def save(self):
        '''save the config to disk'''
        with open(self.ini_file, 'w') as tmp:
            self._config.write(tmp)


def iniexists(file):
    '''(str) -> bool
    Checks if the file exists
    '''
    return _iolib.file_exists(file)
