# pylint: disable=W0612, E0611,E1101
'''pyqt wrapper'''
import sys as _sys
from PyQt4.QtGui import QWidget as _QWidget
from PyQt4.QtGui import QApplication as _QApplication
from PyQt4.QtGui import QMessageBox as _QMessageBox
from PyQt4 import QtCore as _QtCore


def question(title, msg, default_button=_QMessageBox.No, *flags):
    '''(str, str, ints)->QMessageBoxValueEnumeration
    Show a yes/no message box. Flags (ints) are binary ORed to get Yes, No, Ok etc.

    title:
        title of the message box
    msg:
        The message to display

    Returns:
        the result (Eg QMessageBox.Yes)

    Example:
    >>>x = question('Quit this', 'Press yes to quit', _QMessageBox.No, _QMessageBox.Yes)
    print('yes' if x == _QMessageBox.Yes else 'no')
    '''
    options = _or_flags(flags)
    a = _QApplication(_sys.argv)
    w = _QWidget()
    w.setWindowFlags(_QtCore.Qt.WindowStaysOnTopHint)
    return _QMessageBox.question(w, title, msg, options, default_button)


def _or_flags(flags):
    '''ORs a list of values'''
    b = 0
    for x in flags:
        b = b | x
    return b
