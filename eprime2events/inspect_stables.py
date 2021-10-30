#!/usr/bin/python3

import chardet
import csv
from io import BytesIO, StringIO
import os
import pandas as pd
import string
import sys
import zipfile
from pathlib import Path
import loadutils as lu
from get_bytes import get_bytes
# from unitxt import unitxt
from string import capwords
from collections import Counter
from collections.abc import Callable, Iterable, Sequence 
from pandas import DataFrame as df
from typing import Union
from typing import Union
from unidecode import unidecode

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
                        os.PathLike, object]
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
    return result

def check_inpt(inpt:Union[str,os.PathLike,bytes,bytearray]
              ) -> Union[str,bytes]:
    """ Returns appropriate inpt parameter for detection. """
    if bool(isinstance(inpt,(bytes,bytearray))
            or os.path.exists(inpt)):
        inpt = get_bytes(inpt)
    else:
        inpt = inpt        
    datatype = type(inpt)
    return inpt, datatype

def get_lineterm(inpt: Union[bytes, bytearray, str,
                             os.PathLike]
                ) -> Union[str,bytes]:
    """ Returns line end indicator of a file. """
    inpt, datatype = check_inpt(inpt)
    return tuple(pd.Series(tuple(itm[0].strip(itm[1]) for itm in
                                 tuple(zip(inpt.splitlines(keepends=True),
                                           inpt.splitlines()))),
                           dtype=datatype).unique())

def get_separator(inpt, encod:str=None):
    """ Returns delimiter used in a tabulated data file. """
    import re
    inpt, datatype = check_inpt(inpt)
    encod = [def_enc(inpt) if encod is None else encod][0]
    possible=[' ', '\t', '\n', '\r', '\x0b',
              '\x0c',',', ';', ':', '\r\n', r'\|']
    if isinstance(inpt, bytes):
        possible = [itm.encode(encod) for itm in possible]
    pats = tuple(re.compile(pat) for pat in list(possible))
    lineterm = get_lineterm(inpt)
    matches = df(tuple((pat.pattern,len(pat.findall(inpt)))
                     for pat in pats), dtype=datatype)
    delim = matches[0].where(matches[1]==matches[1].max()).dropna().tolist()[0]
    dcount = matches[1].where(matches[1]==matches[1].max()).dropna().tolist()[0]
    if delim in (' ', b' '):
        if len(inpt.split()) != len(inpt.split(delim)):
            delim = ['\s+' if isinstance(inpt, str) else b'\s+']
    return delim

s2sq = lambda s: [[s] if isinstance(s, str) else [s]][0]

def get_znames(zfile:zipfile.ZipFile,
               exts:Union[str, Sequence]='.txt',
               to_exclude:Union[str, Sequence]=badnames,
               to_include:Union[str, Sequence]=None,
              ) -> list:
    """ Returns desired file names from zipfile archive. """
    znames=zfile.namelist()
    if exts is not None:
        exts = s2sq(exts)
        znames = lu.filterlist_inc(include=exts, lst=znames)
    if to_exclude is not None:
        znames = lu.filterlist_exc(exclude=to_exclude,
                                    lst=znames)
    if to_include is not None:
        znames = lu.filterlist_inc(include=to_include,
                                   lst=zfile.namelist())
    return znames

def load_subtables(zdir:Union[str,os.PathLike],
                   exts:Union[str, Sequence]='.txt',
                   to_exclude:Union[str, Sequence]=badnames,
                   to_include:Union[str, Sequence]=prefixes,
                   nfiles:int=None
                  ) -> tuple:
    enclst = []
    for item in os.listdir(zdir):
        zf = zipfile.ZipFile(os.path.join(zdir, item), mode='r')
    
        znames = get_znames(zf, exts=exts, to_exclude=to_exclude,
                            to_include=to_include)
        buffs = tuple(BytesIO(zf.read(name))
                      for name in znames)
        zf.close()
        encs = tuple((def_enc(buff.read()),
                      buff.seek(0))[0]
                     for buff in buffs)
        delims = tuple((get_separator(abuff.read()),
                        abuff.seek(0))[0]
                       for abuff in buffs)
        nrows = tuple((len(abuff.read().splitlines()),
                       abuff.seek(0))[0]
                      for abuff in buffs)

        enclst.append((item, znames, len(znames), encs, delims, nrows))
    results = df(enclst, dtype='string',
                 columns=['archives', 'filenames', 'nfiles',
                          'encodings', 'delimiters', 'nrows'])

    results['is_valid'] = tuple(row[1]['nfiles']==str(nfiles)
                                for row in results.iterrows())

    return results

