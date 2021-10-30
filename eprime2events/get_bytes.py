#!usr/bin/env/python3

import os
from typing import Union
from pathlib import Path
import sys

def get_bytes(inpt: Union[bytes, bytearray, str, os.PathLike, object]):
    """ Returns a bytes object from 'inpt', no matter what 'inpt' is.
        Description
        -----------
        If inpt is a bytes (any kind) object:
            Returns inpt as is;
        If inpt is a string (which is not a file or directory path)
            Returns a bytes (UTF-8) version of inpt;
        If inpt is a path pointing to a file:
            Returns the bytes contained in that file
        *If the return value is a single or empty-byte object,
         returns b"1" (utf8 bytes representation of the number one) """
    if isinstance(inpt, (bytes, bytearray)):
        return inpt
    if os.path.isfile(inpt):
        return Path(inpt).read_bytes()
    if isinstance(inpt, str):
        return inpt.encode(sys.getdefaultencoding())
    else:
        return print("unsupported input type")

def main():
    get_bytes(sys.argv[1])
        
if __name__ == '__main__':
    main()
