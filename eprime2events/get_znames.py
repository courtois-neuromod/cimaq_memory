#!/usr/bin/python3

import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from collections import Counter
from collections.abc import Sequence
from docstring_parser import parse as dsparse
from typing import NewType, Union
from zipfile import ZipFile

exclude = lambda exc, l: [i for i in l if all(s not in i for s in exc)]
include = lambda inc, l: [i for i in l if any(s in i for s in inc)]
s2sq = lambda s: [[s] if isinstance(s, str) else list(s)][0]



def get_znames(zfile: Union[str, os.PathLike, ZipFile],
               exts:Union[str, Sequence]=None,
               to_include:Union[str, Sequence]=None,
               to_exclude:Union[str, Sequence]=None,
               min_size: int = 101,
               nfiles: int = None,
               default_filtering: bool = True,
               keep_open: bool = False,
               prnt: bool = False
              ) -> list:
    """
    Returns desired file names from zipfile archive.

    Lists the contents of a .zip archive. User can filter what
    names are returned by specifying desired file extensions,
    what names to include or exclude, a minimum file size, and what
    number of file names is expected to be returned.
    Default minimum size is 103 bits, which corresponds to a file containing a
    single maximum possible lenght UTF-8 encoded character.
    Due to cross-platform differences, some files thats are not actual members
    of the archive might have been generated and added to its contents
    during compression. These behave as if they were regular archives,
    but are error prone, sometimes unreadable or empty.
    The default behavior is to exclude such files.

    Args:
        zfile: str, os.PathLike, ZipFile
            Path of the zip archive. Can be a regular string,
            a path-like object or an in-memory ZipFile object.
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
        default_filtering: bool, default True
            Exclude commonly error-prone files from the returned names.
            These are ['.DS_Store', '_MACOSX', 'textClipping'], which are
            generally empty.
        keep_open: bool, default False
            Indicate wheter or not to close the zip archive when done.
    """
    badnames = ['.DS_Store', '_MACOSX', 'textClipping']
    bydate = lambda dc: list(dc.keys())[list(dc.values()).index(nfiles)]
    zfile = [zfile if isinstance(zfile, ZipFile)
             else ZipFile(zfile, 'r')][0]
    znames=zfile.namelist()
    # return znames
#     filterlist = exts, to_exclude, to_include
#     exts, to_exclude, to_include = [s2sq(item) if item is not None
#                                     else None for item in filterlist]
    if exts is not None:
        znames = include(exts, znames)
    if to_include is not None:
        znames = include(to_include, znames)
    if to_exclude is not None:
        znames = exclude(to_exclude, znames)
    if default_filtering is True:
        znames = exclude(badnames, znames)
    if min_size is not None:
        znames = [name for name in znames if
                  zfile.getinfo(name).file_size > min_size]
# # for archives with more than nfiles files
    if nfiles is not None:
        if len(znames) > nfiles:
            zinfos = [zfile.getinfo(name) for name in znames]
            dates = [zinfo.date_time[:3] for zinfo in zinfos]
            date_count = dict(Counter(dates))
            try:
                common_date = bydate(date_count)
                znames = [itm[0] for itm in
                          tuple(zip(znames, zinfos, dates))
                          if itm[2] == common_date]
            except ValueError:
                znames = znames
#
    if keep_open is False:
        zfile.close()
    if prnt is True:
        print(znames)
    return znames

def main():
    parser = ArgumentParser(prog=get_znames,
                            usage=get_znames.__doc__,
                            description=get_znames.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('zfile', nargs=1)
    parser.add_argument('--exts', dest='exts', nargs='*',
                        default=None, required=False)
    #                     # action='append')
    parser.add_argument('-i', '--include', dest='to_include',
                        nargs='*', default=None, required=False)
                        # action='append')
    parser.add_argument('-e', '--exclude', dest='to_exclude',
                        nargs='*', default=None, required=False)
    #                     # action='append')
    parser.add_argument('-m', '--min-size', dest='min_size',
                        type=int, nargs='?', default=101, required=False)
    parser.add_argument('-n', '--nfiles', dest='nfiles', type=int,
                        default=None, nargs='?', required=False)
    parser.add_argument('--no-filter', dest='default_filtering',
                        action='store_false')
    parser.add_argument('--keep-open', dest='keep_open', action='store_true')
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    get_znames(args.zfile[0], args.exts, args.to_include, args.to_exclude,
               args.nfiles, args.min_size, args.default_filtering,
               args.keep_open, args.prnt)

if __name__ == '__main__':
    main()
