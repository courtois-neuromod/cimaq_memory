#!/usr/bin/python3

import os
import pandas as pd
import re
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from argparse import MetavarTypeHelpFormatter
from docstring_parser import parse as dsparse
from io import StringIO
from typing import NewType, Union
from unidecode import unidecode

from get_bytes import get_bytes
from get_encoding import get_encoding
from get_has_header import get_has_header
from get_separator import get_separator

hdrz = lambda h: [None if h is False else 0][0]
unibuff = lambda d, enc: StringIO(unidecode(d.decode(enc)))

def read_data(inpt,
              encoding:str=None,
              sep:Union[str, bytes, bytearray]=None,
              file_header: Union[str, int, type(None)]=None,
              prnt: bool=False
             ) -> pd.DataFrame:
    """ Returns a DataFrame with contents from file, string or bytes.

    This function is useful when the data to be converted to a DataFrame
    is of unknown nature, possibly inhomogeneous or made using
    various (possibly unknown a priori) character encoding.
    The data is examined to determine it's encoding, delimiter and
    if it has a file_header. The results are passed to pandas read_fwf or
    read_csv functions according to what delimiter it has.

    Args:
        inpt : bytes, bytearray, str, os.PathLike
            The data to be read.
        encoding : str, optional
            Character encoding of the file.
            Automatically detected if not provided.
        sep : str, bytes, bytearray, optional
            Delimiter used to divide the file sections.
            Automatically detected if not provided.
        prnt : bool, optional
            Print output to stdout.

    Returns: pandas.DataFrame
        The data, read and parsed properly.
    """

    inpt = get_bytes(inpt)
    enc = get_encoding(inpt)
    sep = get_separator(inpt).decode(enc)
    header = hdrz(get_has_header(inpt))
    buff = StringIO(unidecode(inpt.decode(enc)))
    table_params = dict(filepath_or_buffer=buff, sep=sep,
                        header=header, engine='python')
    table = [pd.read_fwf(**table_params) if
             table_params['sep']=='\s+' else
             pd.read_csv(**table_params)][0]
    table = table.dropna(axis=1, how='all')

    # encoding = [encoding if encoding is not
    #             None else get_encoding(inpt)][0]
    # sep = [sep if sep is not None else
    #        get_separator(inpt, encoding)][0]
    # file_header = [file_header if file_header is not None else
    #                hdrz(get_has_header(inpt, encoding))][0]
    # data = [pd.read_fwf(unibuff(inpt, encoding),
    #                     engine='python',
    #                     sep=sep.decode(encoding),
    #                     header=file_header) if sep == '\s+' else
    #         pd.read_csv(unibuff(inpt, encoding),
    #                     engine='python',
    #                     sep=sep.decode(encoding),
    #                     header=file_header)][0]
    if prnt is True:
        print(table)
    return table

def main():
    InptType = NewType('InptType', [bytes, bytearray, str, os.PathLike])
    SepType = NewType('SepType', [str, bytes, bytearray])
    HeaderType = NewType('HeaderType', Union[str, int, type(None)])
    help_messages = [vars(param)['description'] for param in
                     vars(dsparse(read_data.__doc__))['meta']]

    parser = ArgumentParser(prog=read_data,
                            usage=read_data.__doc__,
                            description=read_data.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('inpt', nargs=1, type=InptType, help=help_messages[0])
    parser.add_argument('-e', '--encoding', dest='encoding', required=False,
                        default=None, nargs='?', help=help_messages[1])
    parser.add_argument('-s', '--sep', dest='sep', required=False,
                        default=None, type=SepType, nargs='?',
                        help=help_messages[2])
    # parser.add_argument('-f', '--file_header', dest='file_header',
    #                     required=False, default=None, type=HeaderType,
    #                     nargs='?', help=help_messages[3])
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help=help_messages[-1])
    args = parser.parse_args()
    read_data(args.inpt[0], args.encoding, args.sep,
              args.file_header, args.prnt)

if __name__ == '__main__':
    main()
