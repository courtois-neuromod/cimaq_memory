#!/usr/bin/python3

import os
import pandas as pd
import re
import typing
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pandas import DataFrame as df
from typing import NewType, Union

from get_bytes import get_bytes
from get_encoding import get_encoding

def get_separator(inpt:Union[bytes, bytearray, str, typing.io,
                             os.PathLike, object],
                  encoding:str=None,
                  prnt:bool=False):

    """ Returns delimiter used in a tabulated data file. """

    datatype = type(inpt)
    inpt = get_bytes(inpt)
    encoding = [encoding if encoding is not None
                else get_encoding(inpt)][0]
    possible=[' ', '\t', '\x0b', '\x0c', r'\|', r'\\t', ',', ';', ':']
    possible = [itm.encode(encoding) for itm in possible]
    pats = tuple(re.compile(pat) for pat in list(possible))
    matches = df(tuple((pat.pattern,len(pat.findall(inpt)))
                     for pat in pats),
                 dtype=bytes)
    delim = matches[0].where(matches[1]==matches[1].max()).dropna().tolist()[0]
    if delim in (' ', b' '):
        if len(inpt.split()) != len(inpt.split(delim)):
            delim = ['\s+' if isinstance(inpt, str) else b'\s+'][0]
    if type(delim) != datatype:
        delim = delim.decode(encoding)
    if prnt is True:
        print(str([repr(delim.decode(encoding)) if isinstance(delim, bytes)
               else repr(delim)][0]))
    return delim

def main():
    InptType = NewType('InptType', [bytes, bytearray, str,
                                    os.PathLike, object])
    parser = ArgumentParser(prog=get_separator,
                            usage=get_separator.__doc__,
                            description=get_separator.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('inpt', nargs=1,
                        type=InptType)
    parser.add_argument('-e', '--encoding', dest='encoding', required=False,
                        default=None, type=str, nargs='?',
                        help='Character encoding of the file.')
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    get_separator(args.inpt[0], args.encoding, args.prnt)

if __name__ == '__main__':
    main()
