#!/usr/bin/python3

import os
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Union
from zipfile import ZipFile

from get_default_args import get_default_args
from get_desc import get_desc
from get_znames import get_znames

def get_zinfos(src:Union[str,os.PathLike, ZipFile],
               znames:Union[tuple, list, set]=None,
               exts:Union[str, Sequence]=None,
               to_include:Union[str, Sequence]=None,
               to_exclude:Union[str, Sequence]=None,
               min_size: int = 101,
               nfiles: int = None,
               def_filt: bool = True,
               keep_open: bool = False,
               prnt: bool = False,
               **kwargs) -> list:
    """
    Return desired ZipInfo objects contained in src.

    Works similarly as ``get_znames``, but returns a zip archive member's
    ZipInfo objects instead of its name.

    Args:
        src: str, os.PathLike
            Path of a zip archive
        znames: list, tuple or set, optional
            User-defined list of members contained in the zip archive.
            If not provided, znames will be obtained using ``get_znames``.
            Raises ``KeyError`` if a name is not contained in the archive.
        exts: str, set of str, Sequence of str or None, optional
            File extensions to look for within the archive. Can be a single
            string, a set or a Sequence. Single strings are inserted in a
            list, other types are converted into a list.
        to_include: str, set of str, Sequence of str  or None, optional
            Patterns or names that should be found in the names of the returned
            list. Can be a single string, a set or a Sequence. Single strings
            are inserted in a list, other types are converted into a list.
        to_exclude: str, set of str, Sequence of str  or None, optional
            Patterns or names that should NOT be found in the names of
            the returned list. Can be a single string, a set or a Sequence.
            Single strings are inserted in a list, other types are
            converted into a list.
        min_size: int, default 103
            Minimum required archive size for its name to be returned.
            The default value of 101 corresponds to a file containing a
            single maximum possible lenght UTF-8 encoded character.
            A value of 0 means no minimum size is required.
        nfiles: int, optional
            Expected number of file names to be returned. Used to attempt
            to retrieve nfiles having a common date time if the returned
            number of file names exceeds nfiles file.
        def_filt: bool, default True
            Exclude commonly error-prone files from the returned names.
            These are ['.DS_Store', '_MACOSX', 'textClipping'], which are
            generally empty.
        keep_open: bool, default False
            Indicate wheter or not to close the zip archive when done.
        prnt: bool, default True
            Print output to stdout.
        kwargs:
            Keyword arguments passed to ``get_znames``.
            Valid keywords are the names of the parameters listed above.

    Returns: list
        List of ``ZipInfo`` objects which are members contained
        within the zip archive.
    """

    from get_znames import get_znames
    zfile = [src if isinstance(src, ZipFile)
             else ZipFile(src, 'r')][0]
    znames = [list(znames) if znames is not None else
              get_znames(zfile, exts=exts, to_include=to_include,
                         to_exclude=to_exclude, min_size=min_size,
                         nfiles=nfiles, def_filt=def_filt,
                         keep_open=True)][0]
    zinfos = [zfile.getinfo(zname) for zname in znames]
    if keep_open is False:
        zfile.close()
    if prnt is True:
        print(zinfos)
    return zinfos

def main():
    desc, help_msgs = get_desc(get_zinfos)
    parser = ArgumentParser(prog=get_zinfos,
                            usage=get_zinfos.__doc__,
                            description=desc)
    parser.add_argument('src', nargs=1, help=help_msgs[0])
    parser.add_argument('-z', '--znames', dest='znames', nargs='*',
                        help=help_msgs[1])
    parser.add_argument('--exts', dest='exts', nargs='*')
    parser.add_argument('-i', '--include', dest='to_include', nargs='*')
    parser.add_argument('-e', '--exclude', dest='to_exclude', nargs='*')
    parser.add_argument('-m', '--min-size', dest='min_size', type=int,
                        nargs='?', default=101)
    parser.add_argument('-n', '--nfiles', dest='nfiles', type=int, nargs='?')
    parser.add_argument('--no-filter', dest='def_filt', action='store_false')
    parser.add_argument('--keep-open', dest='keep_open', action='store_true')
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    get_zinfos(args.src[0], args.znames, args.exts, args.to_include,
               args.to_exclude, args.nfiles, args.min_size, args.def_filt,
               args.keep_open, args.prnt)

if __name__ == '__main__':
    main()
