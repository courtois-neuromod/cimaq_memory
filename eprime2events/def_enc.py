#!/usr/bin/python3

import argparse
import chardet
import os
import sys

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from collections import Counter
from pathlib import Path
from typing import Union, NewType

from get_bytes import get_bytes

def get_most_common_enc(inpt: Union[bytes, bytearray, str,
                                    os.PathLike, object]
                       ) -> str:
    """ Returns most common linewise character encoding. """

    inpt = get_bytes(inpt)
    encs = tuple(chardet.detect(line)['encoding']
                 for line in inpt.splitlines(keepends=True))
    result = tuple(dict(Counter(encs).most_common(1)).keys())[0]
    if result == 'ascii':
        result = 'UTF-8'
    return result

def get_inc_enc(inpt: Union[bytes, bytearray, str,
                            os.PathLike, object]
               ) -> str:
    """ Returns char encoding using chardet.UniversalDetector. """

    inpt = get_bytes(inpt)
    detector = chardet.UniversalDetector()
    for line in inpt.splitlines():
        detector.feed(line)
        if vars(detector)['done'] == True:
            break
        detector.close()
        rezz = detector.result
        if rezz['encoding'] == 'ascii':
            rezz.update({'encoding': 'UTF-8'})
        return rezz['encoding']

def def_enc(inpt: Union[bytes, bytearray, str,
                        os.PathLike, object],
            prnt: bool = False
           ) -> str:
    """ Tries several times to detect char encoding. """

    inpt = get_bytes(inpt)
    try:
        result = sys.getdefaultencoding()
        inpt.decode(result)
    except (UnicodeDecodeError, TypeError):
        result = get_most_common_enc(inpt)
        try:
            inpt.decode(result)
        except (UnicodeDecodeError, TypeError):
            result = chardet.detect(inpt)['encoding']
            try:
                inpt.decode(result)
            except (UnicodeDecodeError, TypeError):
                result = get_inc_enc(inpt)
    if prnt is True:
        print(result)
    return result

def main():
    InptType = NewType('InptType', [bytes, bytearray, str,
                                    os.PathLike, object])
    parser = ArgumentParser(prog=def_enc,
                            usage=def_enc.__doc__,
                            description=def_enc.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('inpt', nargs=1,
                        type=InptType)
    parser.add_argument(*('-p', '--print'), dest='prnt',
                        action='store_true',
                        help='print output to stdout')
    args = parser.parse_args()
    def_enc(args.inpt[0], args.prnt)

if __name__ == '__main__':
    main()
