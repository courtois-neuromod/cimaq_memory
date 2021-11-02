#!/usr/bin/python3

import os
import pandas as pd
import sys
import typing
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from collections import Counter
from pandas import DataFrame as df
from typing import NewType, Union

from get_encoding import get_encoding
from get_bytes import get_bytes

InptType = NewType('InptType', [bytes, bytearray, str, typing.io,
                                os.PathLike, object])

def get_lineterm(inpt: InptType,
                 encoding: str = None,
                 prnt: bool = False
                ) -> Union[str,bytes]:
    """ Returns line end indicator of a file. """

    datatype = type(inpt)
    inpt = get_bytes(inpt)
    encoding = [encoding if encoding is not None
                else get_encoding(inpt)][0]
    lterm = Counter(tuple(itm[0].strip(itm[1]) for itm in
                  tuple(zip(inpt.splitlines(keepends=True),
                            inpt.splitlines())))).most_common(1)[0][0]
    if type(inpt) != datatype:
        lterm = lterm.decode()
    if prnt is True:
        print(str([repr(lterm.decode(encoding)) if isinstance(lterm, bytes)
               else repr(lterm)][0]))
    return lterm

def main():
    parser = ArgumentParser(prog=get_lineterm,
                            usage=get_lineterm.__doc__,
                            description=get_lineterm.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('inpt', nargs=1,
                        type=InptType)
    parser.add_argument(*('-e', '--encoding'), dest='encoding',
                        default=None, required=False, nargs='?',
                        help='Character encoding of the file.')
    parser.add_argument(*('-p', '--print'), dest='prnt',
                        action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    get_lineterm(args.inpt[0], args.encoding, args.prnt)

if __name__ == '__main__':
    main()
