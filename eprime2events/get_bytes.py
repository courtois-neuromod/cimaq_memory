#!usr/bin/env/python3

import os
import typing
from typing import NewType, Union
from pathlib import Path
import sys

def get_bytes(inpt: Union[bytes, bytearray, str, os.PathLike,
                          typing.io, object]) -> bytes:
    """
    Returns a bytes object from 'inpt', no matter what 'inpt' is.

    If inpt is a buffer object, its contents is read.
    If the read input is bytes, it is returned as is,
    if it is text, it is encoded to the system's default character
    encoding value. Other types of input are treated as follows:
    If inpt is a bytes (any kind) object:
        Returns inpt as is;
    If inpt is a string (which is not a file or directory path)
        Returns a bytes (UTF-8) version of inpt;
    If inpt is a path pointing to a file:
        Returns the bytes contained in that file

    Args:
        inpt: bytes, bytearray, str, os.PathLike, typing.io, object
            The object to get the bytes representation from.

    Returns: bytes
    """

    if hasattr(inpt, 'read'):
        inpt = inpt.read()
    if isinstance(inpt, (bytes, bytearray)):
        return inpt
    if os.path.isfile(inpt):
        return Path(inpt).read_bytes()
    if isinstance(inpt, str):
        return inpt.encode(sys.getdefaultencoding())
    else:
        return print("unsupported input type")

def main():
    from argparse import ArgumentParser
    InptType = NewType('InptType', [bytes, bytearray, str, os.PathLike,
                                    typing.io, object])
    parser = ArgumentParser(prog=get_bytes,
                            usage=get_bytes.__doc__,
                            description=get_bytes.__doc__.splitlines()[0])
    parser.add_argument('inpt', type=InptType, nargs=1)
    args = parser.parse_args()
    get_bytes(args.inpt[0])

if __name__ == '__main__':
    main()
