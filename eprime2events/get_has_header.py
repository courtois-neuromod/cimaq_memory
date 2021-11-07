#!usr/bin/env/python3

import os
import pandas as pd
import sys
import typing
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from io import BytesIO
from typing import NewType, Union
from unidecode import unidecode

from get_desc import get_desc
from get_encoding import get_encoding
from get_bytes import get_bytes

def get_has_header(
    inpt: Union[bytes, bytearray, str, os.PathLike, typing.io,
                pd.DataFrame, object],
    encoding: str = None,
    prnt: bool = False
) -> bool:
    """ Returns True if 1st line of inpt is a header line

    Args:
        inpt: Bytes, string, file path, in-memory buffer
              Object to inspect
            - See help(get_bytes)
        encoding: bool, optional
                  Character encoding of the object to inspect
        prnt: bool

    Returns: bool
             True if the object's first row is a header, otherwise False
    """
    if isinstance(inpt, pd.DataFrame):
        enc = sys.getdefaultencoding()
        inpt = unidecode('\n'.join(['\t'.join([str(i) for i in line])
                         for line in inpt.values.tolist()])).encode(enc)
    inpt = get_bytes(inpt)
    encoding = [encoding if encoding is not None else get_encoding(inpt)][0]
    got_hdr = [bool(BytesIO(inpt).read(1)
               not in bytes(".-0123456789", encoding))
               if len(inpt.splitlines()) > 1 else False][0]
    if prnt is True:
        print(got_hdr)
    return got_hdr

def main():
    desc, help_msgs = get_desc(function_name=get_has_header)
    InptType = NewType('InptType', [bytes, bytearray, str, typing.io,
                                    os.PathLike, pd.DataFrame, object])
    parser = ArgumentParser(prog=get_has_header,
                            usage=get_has_header.__doc__,
                            description=get_has_header.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('inpt', nargs=1, type=InptType, help=help_msgs[0])
    parser.add_argument('-e', '--encoding', dest='encoding', required=False,
                        default=None, type=str, nargs='?',
                        help=help_msgs[1])
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help=help_msgs[-1])
    args = parser.parse_args()
    get_has_header(args.inpt[0], args.encoding, args.prnt)

if __name__ == '__main__':
    main()
