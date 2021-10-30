#!usr/bin/bash

from argparse import ArgumentParser
# from argparse import HelpFormatter
from argparse import ArgumentDefaultsHelpFormatter
import chardet
from io import BytesIO
from io import StringIO
import os
import pandas as pd
import loadutils as lu
import zipfile
from typing import Union
from sniffbytes import get_bencod
from unitxt import unitxt
import importlib
from os.path import expanduser as xpu
import sys

#__import__('sniffbytes-0')) as sniffbytes

def get_taskfiles(src:Union[str,os.PathLike],
                  dst:Union[str,os.PathLike]=None,
                  **kwargs)->None:
    """ Extracts unicode-compliant event files from archive.

    Extracts only the necessary files from each participant's
    archive. The files are formatted to be unicode-compliant
    and can be read as regular tsv files.

    Args:
        src:
            Participant's zipfile archive.
        dst (optional):
            Desired output directory where to
            extract the task files.
            Defaults to thw current working directory.
    Returns:
        None

    Notes:
        Taskfiles are written to dst within a
        sub-directory corresponding to the
        participant's id, similar to the
        BIDs structure.
    """

    prefix = ['Output-Responses-Encoding_CIMAQ_',
              'Onset-Event-Encoding_CIMAQ_',
              'Output_Retrieval_CIMAQ_']

    sub_id = '-'.join(['sub', os.path.basename(src).split('_')[0]])
    v_num = os.path.basename(src).split('_')[1]
    fnames = tuple('_'.join([sub_id, v_num, prf])+'.tsv'
                   for prf in prefix)
    dst = [dst if dst is not None else
           os.path.join(os.getcwd(), sub_id)][0]
    os.makedirs(os.path.join(dst, v_num), exist_ok=True)

    with zipfile.ZipFile(src, mode='r') as subzip:
        sfnames = lu.filterlist_inc(include=prefix,
                                    lst=subzip.namelist())
        if len(sfnames) != 3:
#            break
            print(f'{sub_id} has {len(sfnames)} files')
            sys.exit()

        buffs = tuple(BytesIO(subzip.read(afile))
                      for afile in sfnames)
        subzip.close()
    encods = tuple((get_bencod(abuff.read()),
                    abuff.seek(0))[0]
                   for abuff in buffs)

    b2sbuff = lambda buf,enc: StringIO(unitxt(buf.read().decode(enc)))
    str_buffs = tuple(b2sbuff(itm[0], itm[1])
                      for itm in tuple(zip(buffs, encods)))
    filedict = {itm[0]: pd.read_csv(itm[1], sep='\t', engine='python',
                                    header=0, encoding='UTF-8-SIG')
                for itm in tuple(zip(fnames, str_buffs))}
    tuple(itm[1].to_csv(os.path.join(dst, v_num, itm[0]),
                        sep='\t', encoding='UTF-8-SIG')
          for itm in tuple(filedict.items()))

#def get_args():
def main():
    init_dict = dict(prog=get_taskfiles,
                     usage=get_taskfiles.__doc__,
                     description=__name__.__doc__.splitlines()[0],
                     formatter_class=ArgumentDefaultsHelpFormatter)

    _parser = ArgumentParser(**init_dict)
    _parser.add_argument('src', nargs=1)
    _parser.add_argument('-d', '--dst', nargs='?', required=False, default=None)
    args = _parser.parse_args()
    get_taskfiles(args.src[0],args.dst)

if __name__ == '__main__':
    main()

