#!/usr/bin/python3

import argparse
import os
import sys
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from argparse import MetavarTypeHelpFormatter
from typing import Union

def loaddirs(
  src: Union[os.PathLike, str],
  prnt:bool=False
) -> list:
    """ Returns recursive full relative directory paths in src.
        
    Args:
        src: Union[os.PathLike, str]
            Path of directory to be scanned

    Returns:
        dirlist: list
            1D list full relative paths
    """

    dirlist = []
    for allfiles in os.walk(src):
        for adir in allfiles[1]:
            output = os.path.join(allfiles[0], adir)
            if os.path.isdir(output):
                dirlist.append(output)
    if prnt is True:
        print(dirlist)
    return dirlist
    
def main():
    parser = ArgumentParser(prog=loaddirs,
                            usage=loaddirs.__doc__,
                            description=loaddirs.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
#    parser.add_argument('-h', '--help', nargs='?', required=False)
    parser.add_argument('src', nargs=1)
    parser.add_argument(*('-p', '--print'), dest='prnt',
                        action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()    
    loaddirs(args.src[0], args.prnt)

if __name__ == '__main__':
    main()

