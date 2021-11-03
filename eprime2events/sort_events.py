#!/usr/bin/python3

import os
import re
import tempfile
import zipfile

from argparse import ArgumentParser
from os.path import splitext
from pandas import DataFrame as df
from pathlib import Path
from typing import NewType, Sequence, Union
from unidecode import unidecode

from fix_dupindex import fix_dupindex
from get_has_dupindex import get_has_dupindex
from get_znames import get_znames
from read_data import read_data

# prefixes = ['Output-Responses-Encoding_', 'Onset-Event-Encoding_', 'Output_Retrieval_']

def sort_events(zdir: Union[str, os.PathLike],
                sub_id_pattern: str,
                ses_pattern: str,
                dst: Union[str, os.PathLike] = None,
                # exts: Union[str, Sequence] = None,
                to_include: Union[str, Sequence] = None,
                # to_exclude: Union[str, Sequence] = None,
                nfiles: int = None,
                min_size: int = 101):
    """
    Extract participant archive files in a BIDS-compliant fashion.

    The BIDS directory structure is achieved by finding relevant
    identifiers among file names using regular expressions.
    Parameters sub_id_pattern and ses_pattern should be adjusted
    to match participant identifiers and session number denotation.
    Other parameters are passed to get_znames.py to control what files
    are extracted from each archive.

    Args:
        zdir: str, os.PathLike
            The directory containing participants' zip file archives.
        sub_id_pattern: str
            Regular expression string used to find participant identifiers.
        ses_pattern: str
            Regular expression string used to find session identifiers.
        dst: str, os.PathLike, optional
            Directory where to create the BIDS structure and extract
            the archives. If not provided, defualts to the system's
            temporary directory.
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
        nfiles: int, optional
            Expected number of file names to be returned. Used to attempt
            to retrieve nfiles having a common date time if the returned
            number of file names exceeds nfiles file.
        min_size: int, default 101
            Minimum required archive size for its name to be returned.
            The default value of 101 corresponds to a file containing a
            single maximum possible lenght UTF-8 encoded character.
            A value of 0 means no minimum size is required.

    Returns: None
    """

    incomplete_cols = ['participant_id', 'session', 'archive_name',
                       'nfiles', 'contents']
    sub_id_pat = re.compile(sub_id_pattern)
    ses_pat = re.compile(ses_pattern)
    zname_params = dict(to_include=to_include,
                        # to_exclude=to_exclude,
                        min_size=min_size,
                        nfiles=nfiles, keep_open=True)
    dst = [dst if dst is not None else
           os.path.join(tempfile.gettempdir(), 'extracted_dataset')][0]
    os.makedirs(dst, exist_ok=True)
    allinfos, irreg = [], []
    for zfile in enumerate(os.listdir(zdir)):
        with zipfile.ZipFile(os.path.join(zdir, zfile[1]), mode='r') as zf:
            try:
                sub_id = '-'.join(['sub',sub_id_pat.findall(zfile[1])[0]])
            except IndexError:
                sub_id = f'sub-no_id{zfile[1]}'
            try:
                v_num = ses_pat.findall(zfile[1])[0]
            except IndexError:
                v_num = 'VXX'
            znames = get_znames(zf, **zname_params)
            print(len(znames), nfiles, len(znames)==nfiles)
            if len(znames) == int(nfiles):
                # zf.close()
                os.makedirs(os.path.join(dst, sub_id),
                            exist_ok=True)
                os.makedirs(os.path.join(dst, sub_id, v_num), exist_ok=True)
                for zname in znames:
                    new_zname = unidecode(Path(os.path.basename(zname)).resolve().name)
                    fname = '_'.join([sub_id, f'ses-{v_num}',
                                      splitext(new_zname)[0].replace('.', '_')+'.tsv'])
                    table = [fix_dupindex(read_data(zf.read(zname).lower()))
                             if get_has_dupindex(read_data(zf.read(zname)))
                             else read_data(zf.read(zname).lower())][0]
                    table.to_csv(os.path.join(dst, sub_id, v_num, fname),
                                                     sep='\t', encoding='UTF-8-SIG')
            else:
                irreg.append((sub_id, v_num, zfile[1], len(znames), znames))
            zf.close()
    df(((itm for itm in irreg)),
       columns=incomplete_cols).to_csv(os.path.join(dst, 'incomplete_archives.tsv'),
                                       sep='\t', encoding='UTF-8-SIG')

def main():
    from get_desc import get_desc
    desc, help_msgs = get_desc(sort_events)
    FilterType = NewType('FilterType', [str, list, set, tuple, Sequence])
    parser = ArgumentParser(prog=get_znames, usage=get_znames.__doc__,
                            description=desc)
    parser.add_argument('zdir', nargs=1, help=help_msgs[0])
    parser.add_argument('sub_id_pattern', nargs=1, help=help_msgs[1])
    parser.add_argument('ses_pattern', nargs=1, help=help_msgs[2])
    parser.add_argument('-d', '--dst', dest='dst', nargs='?',
                        help=help_msgs[3])
    # parser.add_argument('--exts', dest='exts', nargs='*',
    #                     default=None, required=False)
    #                     # action='append')
    parser.add_argument('-i', '--include', dest='to_include',
                        nargs='*', default=None, required=False)
                        # action='append')
    # parser.add_argument('-e', '--exclude', dest='to_exclude',
    #                     nargs='*', default=None, required=False)
    #                     # action='append')
    parser.add_argument('-n', '--nfiles', dest='nfiles', type=int,
                        default=None, nargs='?', required=False)
    parser.add_argument('-m', '--min_size', dest='min_size',
                        type=int, nargs='?', default=101, required=False)

    args = parser.parse_args()
    sort_events(args.zdir[0], args.sub_id_pattern[0],
                args.ses_pattern[0], args.dst,
                # args.exts,
                args.to_include,
                # args.to_exclude,
                args.nfiles, args.min_size)

if __name__ == '__main__':
    main()
