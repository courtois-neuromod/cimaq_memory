#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import re
import shutil
import tempfile
from argparse import ArgumentParser
from typing import Union
from unidecode import unidecode
from get_desc import get_desc
from load_cimaq_taskfiles import load_cimaq_taskfiles

def ig_f(adir, files):
    return [f for f in files if
            os.path.isfile(os.path.join(adir, f))]

def findbyvalues(matchdict, table):
    from loadutils import flatten
    table = table.copy(deep=True).fillna('NA').astype(str)
    patterns = [re.compile(pattern+'|NA', re.I)
                for pattern in tuple(matchdict.values())]
    names = tuple(matchdict.keys())
    matchlist = []
    [matchlist.append([col[1] for col
                      in enumerate(table.columns)
                      if table[col[1]].str.match(pat).all()])
     for pat in patterns]
    matchcols = sorted(set(flatten(matchlist)))
    renamer = dict(tuple(zip(matchcols, names)))
    return renamer

class NamedMatrix():
    def __init__(self, path, name=None, dirname=None,
                 table=None):
        from pathlib import Path
        from read_data import read_data
        self.path = Path(path)
        self.name = [name if name is not None else
                     os.path.basename(path)][0]
        self.dirname = Path(os.path.dirname(path))
        self.table = read_data(path)

src = '/home/francois/cimaq_eprime_conversion/'
dst = '/home/francois/COPYTREETEST/'

def make_cimaq_encoding_retrieval(src:Union[str,os.PathLike],
                                  dst:Union[str,os.PathLike],
                                  matchdict:dict=None
                                  ) -> None:
    """
    Create proper CIMA-Q encoding and retrieval task files.

    From the split, yet sorted CIMA-Q fMRI and behavioural task files,
    (after data integrity check, see ``sort_events.py``),
    create the correct files and sort in a BIDS-compliant directory
    structure.

    Args:
        src: str, os.PathLike
             Input directory, where files generated by ``sort_events``
             were saved.
        dst: str, os.PathLike
            Directory where to create the BIDS structure and extract
            the archives. If not provided, defualts to the system's
            temporary directory.
        matchdict: dict
            Dictionary containing the stimuli onsets file column names
            as keys and their corresponding regex pattern strings as values.

    Returns: None
    """

    mdict = {'trialnumber': '\d+', 'category': '\w{3}',
             'trialcode': '\w{3}\d{1,}',
             'oldnumber': '\w{3}\d{2}',
             'condition': '[control|encoding]',
             'onset': '\d+\.\d+',
             'duration': '\d+\.\d+',
             'fixation': 'fixation',
             'offset': '\d+\.\d+',
             'isi': '\d+\.\d+'}

    dst = [dst if dst is not None else
           os.path.join(tempfile.gettempdir(), 'cimaq_task_files')][0]
    matchdict = [matchdict if matchdict is not None else mdict][0]
    taskpaths = [[NamedMatrix(apath) for apath in sorted(item)]
                 for item in load_cimaq_taskfiles(src)]
    shutil.copytree(src, dst, ignore=ig_f)
    newitems = []

    for item in taskpaths:
        sub_id, ses_num = item[0].name.split('_')[:2]

        onsf = item[0].table.copy(deep=True)
        encf = item[1].table.copy(deep=True)
        encf = encf[['trialnumber','category', 'oldnumber',
                     'correctsource', 'stim_rt']]
        encf['correctsource'] = encf['correctsource'].fillna(0.0)
        retf = item[-1].table.copy(deep=True)
        onsf = onsf.rename(findbyvalues(matchdict=matchdict,
                                        table=onsf), axis=1)
        onsf = onsf[['onset', 'duration','offset','isi']]
        encf = pd.concat([encf, onsf], axis=1)
        encf['offset'] = encf['offset'].astype(float)
        encf['onset'] = encf['onset'].astype(float)
        encf['duration'] = encf['offset']-encf['onset']
        encf['stim_rt'] = encf['stim_rt'].astype(float).div(1000)

        retf['position_correct'] = retf.oldnumber.map(dict(zip(encf.oldnumber,
                                                               encf.correctsource)))
        retf['spatial_resp'] = [[row[1]['spatial_resp'] if
                                 row[1]['recognition_resp'] == 1
                                 else 0.0][0] for row in retf.iterrows()]
        retf['position_correct'] = retf['position_correct'].fillna(0.0)
        retf = retf[['stim', 'oldnumber', 'recognition_resp',
                     'recognition_rt', 'spatial_resp',
                     'spatial_rt', 'position_correct']]

        retf['recognition_acc'] = [bool(row[1]['recognition_resp']==2
                                         and 'new' in row[1]['oldnumber'])
                                    or bool(row[1]['recognition_resp']==1
                                         and 'old' in row[1]['oldnumber'])
                                    for row in retf.iterrows()]

        retf['spatial_acc'] = [row[1]['spatial_resp']==row[1]['position_correct']
                                if row[1]['position_correct'] != 0.0 else np.nan
                                for row in retf.iterrows()]
        retf['spatial_rt'] = [row[1]['spatial_rt']
                               if row[1]['recognition_resp']==1
                               else np.nan for row in retf.iterrows()]
        retf['spatial_resp'] = retf['spatial_resp'].replace({0.0:np.nan})
        retf['recognition_rt'] = retf['recognition_rt'].astype(float).div(1000)
        retf['recognition_rt'] = [row[1]['recognition_rt'] if
                                  row[1]['recognition_resp'] == 1
                                   else np.nan for row in retf.iterrows()]
        retf['spatial_rt'] = retf['spatial_rt'].astype(float).div(1000)
        retf['position_correct'] = retf['position_correct'].replace({0.0:np.nan})

        encf['stim_file'] = encf.oldnumber.map(dict(zip(retf.oldnumber,
                                                               retf.stim)))

        # Accounting for niquely-named stimulus ``hard_musical_recorder.bmp``
        encf['stim_category'] = [row[1]['stim_file'].replace('hard_',
                                                             '').split('_',
                                                                       maxsplit=1)[0]
                                 if row[1]['stim_file'] is not np.nan else np.nan
                                 for row in encf.iterrows()]

        for col in retf.iloc[:, 3:]:
            encf[col] = encf.oldnumber.map(dict(zip(retf.oldnumber,
                                                        retf[col])))
        encf = encf.drop('position_correct', axis=1)
        encf = encf.replace({'NA', np.nan})
        encf = encf.iloc[int(encf.shape[0]-retf.shape[0]):,:]
        encf = encf.reset_index(drop=True)
        item[-1].__setattr__('behavioural_file', retf)
        item[1].__setattr__('events_file', encf)
        newitems.append(item)

    for item in newitems:
        sub_id, ses_num = item[0].name.split('_')[:2]
        encsuffix = '_'.join([sub_id, ses_num, 'task-memory', 'events'])
        retsuffix = '_'.join([sub_id, ses_num, 'task-memory', 'behavioural'])
        item[1].events_file.to_csv(os.path.join(dst,
                                                sub_id, ses_num.split('-')[1],
                                                encsuffix)+'.tsv',
                                   sep='\t', encoding='UTF-8-SIG', index=None)
        item[-1].behavioural_file.to_csv(os.path.join(dst, sub_id,
                                                      ses_num.split('-')[1],
                                                      retsuffix)+'.tsv',
                                   sep='\t', encoding='UTF-8-SIG', index=None)

def main():
    desc, helps = get_desc(make_cimaq_encoding_retrieval)
    parser = ArgumentParser(prog=make_cimaq_encoding_retrieval,
                            usage=make_cimaq_encoding_retrieval.__doc__,
                            description=desc)
    parser.add_argument('src', nargs=1, help=helps[0])
    parser.add_argument('dst', nargs=1, help=helps[1])
    parser.add_argument('matchdict', nargs='?', type=dict, help=helps[-1])
    args = parser.parse_args()
    make_cimaq_encoding_retrieval(args.src[0], args.dst[0], args.matchdict)

if __name__ == '__main__':
    main()
