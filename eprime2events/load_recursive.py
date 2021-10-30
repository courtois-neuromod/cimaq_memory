#!usr/bin/python3

import argparse
import os
import sys
from typing import Union
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

def load_recursive(src: Union[os.PathLike, str],
                   prnt:bool=False) -> list:
    """ Lists the full relative path files in a directory.

       Args:
           src: type = str
                Name of the directory containing the files.
       Returns:
           flist: type = list
                  List containing all files' full relative paths.
    """

    flist = []
    for allfiles in os.walk(src):
        for afile in allfiles[2]:
            adir = os.path.join(allfiles[0], afile)
            if os.path.isfile(adir):
                flist.append(adir)
    if prnt is True:
        print(flist)
    return flist

def main():
    parser = ArgumentParser(prog=load_recursive,
                            usage=load_recursive.__doc__,
                            description=load_recursive.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('src', nargs=1)
    parser.add_argument(*('-p', '--print'), dest='prnt',
                        action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    load_recursive(args.src[0], args.prnt)

if __name__ == '__main__':
    main()
