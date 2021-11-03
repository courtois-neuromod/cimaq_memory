#!usr/bin/env/python3

import os
import pandas as pd
import sys
import typing
from argparse import ArgumentParser
from typing import NewType, Union
from unidecode import unidecode

from get_bytes import get_bytes
from get_encoding import get_encoding
from get_has_header import get_has_header
########################################################################
eveseq = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 == 0)
oddseq = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 != 0)



def get_has_dupindex(inpt: [bytes, bytearray, str, os.PathLike,
                            pd.DataFrame, object],
                     encoding: str = None,
                     prnt: bool = False) -> bool:
    """
    Returns True or False depending if inpt has a duplicate index

    Returns True if the first item of even and odd lines is repeated.
    Returns False otherwise or upon IndexError.
    As IndexError is raised when trying to access values by an index
    out of a sequence boundairies, IndexError indicates single-byte
    files. Being a single byte, it can't be a duplicate index.
    
    Args:
        inpt: bytes, bytearray, str, os.PathLike, pd.DataFrame or object
            Object to analyse.
        encoding: str, optional
            Character encoding of inpt.
        prnt: bool, optional
            Print output to stdout
    
    Returns: bool
        True if inpt has a duplicate index, otherwise False.

    """
    if isinstance(inpt, pd.DataFrame):
        enc = sys.getdefaultencoding()
        inpt = unidecode('\n'.join(['\t'.join([str(i) for i in line])
                         for line in inpt.values.tolist()])).encode(enc)
    inpt = get_bytes(inpt)
    enc = [encoding if encoding is not None
           else get_encoding(inpt)][0]
    file_header =  get_has_header(inpt, encoding=enc)
    lines = [inpt.splitlines()[1:]
             if file_header else inpt.splitlines()][0]
    ev_items, od_items = eveseq(lines), oddseq(lines)
    try:
#        return ev_items[0].split()[0] == od_items[0].split()[0]
        eve_index = [line.split()[0] for line in ev_items]
        odd_index = [line.split()[0] for line in od_items]
        val = eve_index == odd_index
    except IndexError:
        val = False
    if prnt is True:
        print(val)
    return val


def main():
    from get_desc import get_desc
    desc, help_msgs = get_desc(get_has_dupindex.__doc__)
    parser_args = dict(usage=get_has_dupindex.__doc__,
                       description=desc)
    InptType = NewType('InptType', [bytes, bytearray, str, os.PathLike,
                                    pd.DataFrame, object])
    parser = ArgumentParser(prog=get_has_dupindex, **parser_args)
    parser.add_argument('inpt', nargs=1, type=InptType,
                        help=help_msgs[0])
    parser.add_argument('-e', '--encoding', dest='encoding',
                        required=False, default=None, type=str,
                        nargs='?', help=help_msgs[1])
    parser.add_argument('-p', '--print', dest='prnt',
                        action='store_true', help=help_msgs[-1])
    args = parser.parse_args()
    get_has_header(args.inpt[0], args.encoding, args.prnt)

if __name__ == '__main__':
    main()
    if __name__ == __main__:
        get_has_dupindex(inpt, encoding, has_header)
