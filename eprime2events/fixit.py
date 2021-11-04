#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
import re
import tempfile
import time
from argparse import ArgumentParser
from typing import Union
from pandas import DataFrame as df

from get_desc import get_desc
from load_cimaq_taskfiles import load_cimaq_taskfiles
from load_recursive import load_recursive
from make_cimaq_stim_map import make_cimaq_stim_map

revdict = lambda d: dict(tuple((i[1], i[0]) for i in tuple(d.items())))
get_uids = lambda lst, s=0: dict(tuple(enumerate(sorted(list(lst)), s)))
intersect = lambda lst1, lst2: [v for v in tuple(set(lst1)) if v in set(lst2)]

encoding_cols = ['trial_num','trial_type','pos',
                 'stim_rt','stim_id', 'categ_id','onset',
                 'duration','offset','isi']
retrieval_cols = ['trial_type','recog_rt','recog_resp',
                  'pos_resp', 'pos_rt', 'pos',
                  'stim_id','categ_id']
onset_cols = ['oldnumber','onset', 'offset','isi']

def fixit(src:Union[str, os.PathLike],
          enc_cols:list=encoding_cols,
          ret_cols:list=retrieval_cols,
          ons_cols:list=onset_cols,
          dst:Union[str, os.PathLike]=None) -> None:
    """
    Fix the CIMA-Q task files.

    Repair the CIMA-Q task files by matching trials for the
    Encoding task (2 files) and Retrieval task (1 file)
    on an intact column with common values.
    Non numeric data (stimuli names and categories) is converted
    to numeric by mapping an index on each of these values.
    Timings in miliseconds are converted to seconds.
    Across tasks, the target condition (stimulus to remember)
    is always represented by the number 1 and others (control, new)
    are represented by the number 0 to avoid confusion.
    The resulting tables data type is <float> for all columns.

    Args:
        src: str, os.PathLike
            The top-level directory of the files, sorted
            according to BIDS Standard.
        enc_cols: list, default = encoding_cols
            List of strings representing the names of the resulting
            (repaired) Encoding task file.
        ret_cols: list, default = retrieval_cols
            List of strings representing the names of the resulting
            (repaired) Retrieval task file.
        ons_cols: list, default = onset_cols
            List of strings representing the names of the
            non-redundant columns within the onset file.
        dst: str, os.PathLike, optional
            Destination directory where to write the files.
            Defaults to system-specific temporary directory.

    Returns: None

    Notes:
        Note 0: Button-press
            The button-press indicator column is removed
            as it was not always recorded for each participant.
            The button-press timings were however recorded
            consistantly.
        Note 1: Clean up eprime programming mistake
            Replace spatial_resp and spatial_rt values with NaN if subject
            perceived image as 'new' (the image was not probed for pos).
            There should be no response or RT value there.
            Values were carried over from previous trial (not reset in eprime)
            CONFIRMED w Isabel: subject must give a pos answer when probed
            (image considered OLD) before eprime moves to the next trial.
    """

#     testpaths=load_cimaq_taskfiles(src)
    testpaths = tuple(set(os.path.dirname(fpath) for fpath in load_recursive(src)
                          if os.path.basename(fpath).startswith('V')))
    ustims = make_cimaq_stim_map(src)
    print(ustims)
    dst = [dst if dst is not None else
           os.path.join(tempfile.gettempdir(),
                        os.path.basename(src)+'_fixed')][0]
    os.makedirs(dst, exist_ok=True)

    for item in testpaths:
#         item = sorted(list(item))
        item = tuple(os.path.join(item, itm) for itm in sorted(os.listdir(item)))
        sub_id = os.path.basename(os.path.dirname(os.path.dirname(item[0])))
        v_num = os.path.basename(os.path.dirname(item[0]))
        mid = 'task-memory'
        enc_suf, ret_suf = 'events.tsv', 'behavioural.tsv'
        onsets, enc, ret = [pd.read_csv(itm, sep='\t', dtype=str, engine='python')
                                    for itm in sorted(item)]
        print((onsets, enc, ret))
        # Remove redundant columns
#         try:
        onsets = onsets.drop(['0','1','2','4','6','7'], axis=1)
        onsets = onsets.set_axis(['oldnumber','onset',
                                      'offset','isi'], axis=1)
#         except KeyError:
#           print(os.path.dirname(item[0]))
#           continue
#           pass
        # Note 0: Button-press
        if 'stim_resp' in tuple(enc.columns):
            enc = enc.drop(['stim_resp'], axis=1)

        enc = enc.drop(['stim_acc'], axis=1)
        enc['stim'] = enc.oldnumber.map(dict(zip(ret.oldnumber,
                                                 ret.stim)))
        enc['stim_id'] = enc.stim.map(dict(zip(ustims.stim,
                                               ustims.index)))
        enc['categ_id'] = enc.stim_id.map(dict(zip(ustims.index,
                                                   ustims.categ_id)))
        ret['pos'] = ret.oldnumber.map(dict(zip(enc.oldnumber,
                                                     enc.correctsource)))
        ret['stim_id'] = ret.stim.map(dict(zip(ustims.stim,
                                               ustims.index)))
        ret['categ_id'] = ret.stim.map(dict(zip(ustims.stim,
                                                ustims.categ_id)))

        enc[ons_cols] = onsets[ons_cols]
#         except KeyError:
#           print(os.path.dirname(item[0]))
#           pass
        enc = enc.drop(['trialcode', 'oldnumber', 'stim'],
                               axis=1).fillna('NA')
        enc['category'] = enc['category'].replace({'ctl':'0',
                                                           'enc':'1'})
        ret = ret.fillna('NA')
        ret['recognition_resp'] = ret['recognition_resp'].replace({'0':'NA'})
        ret['category'] = ret['category'].replace({'new':'0', 'old':'1'})
        ret = ret.drop(['stim','oldnumber', 'recognition_acc'], axis=1)
        ret['recognition_resp'] = ret['recognition_resp'].replace({})
        # Convert str to float and ms to s
        enc = enc.replace({'NA': np.nan})
        enc = enc.astype(float)
        enc['stim_rt'] = enc['stim_rt'].div(1000)
        enc.insert(loc=7, column='duration',
                       value=enc['offset']-enc['onset'])
        # Set the new Encoding column names
        enc = enc.set_axis(enc_cols, axis=1)
        enc = enc.set_index(tuple(enc.columns)[0], drop=True)
        # Note 1: Clean up eprime programming mistake
        ret['recognition_resp'] = ret['recognition_resp'].replace({'2':'0'})
        ret[['spatial_resp',
             'spatial_rt']] = [(row[1]['spatial_resp'],
                                row[1]['spatial_rt'])
                                if row[1]['recognition_resp']=='1'
                                else ('NA', 'NA')
                                for row in ret.iterrows()]
        # Set the new Retrieval column names
        ret = ret.set_axis(ret_cols, axis=1)
        ret['recog_acc'] = ret['recog_resp']==ret['trial_type']
        ret['pos_acc'] = ret['pos_resp']==ret['pos']
        # Convert str to float and ms to s
        ret = ret.replace({'NA':np.nan})
        ret = ret.astype(float)
        ret['recog_rt'] = ret['recog_rt'].div(1000)
        ret['pos_rt'] = ret['pos_rt'].div(1000)
        ret = ret.reset_index(drop=False).rename({'index':'trial_num'},
                                                         axis=1)
        ret = ret.set_index('trial_num', drop=True)
        savepath = os.path.join(dst, sub_id, v_num)
        os.makedirs(os.path.join(dst, sub_id), exist_ok=True)
        os.makedirs(os.path.join(dst, sub_id, v_num),
                                 exist_ok=True)
        print((enc, ret))
        enc.to_csv(os.path.join(savepath, '_'.join([sub_id, 'ses-'+v_num,
                                                    mid, enc_suf])),
                   sep='\t', encoding='UTF-8-SIG')
        ret.to_csv(os.path.join(savepath, '_'.join([sub_id, 'ses-'+v_num,
                                                    mid, ret_suf])),
                   sep='\t', encoding='UTF-8-SIG')
        # Fix irregular whitespaces
        ustims = ustims.astype(str)
        for col in ustims.columns:
            ustims[col] = ustims[col].str.replace(' ', '_').str.lower()
        ustims.to_csv(os.path.join(dst, 'cimaq_stimuli.tsv'),
                      sep='\t', encoding='UTF-8-SIG')

def main():
    desc, help_msgs = get_desc(fixit)
    parser = ArgumentParser(prog=fixit, description=desc, usage=fixit.__doc__)
    # parser.set_defaults(**dict(enc_cols=encoding_cols,
    #                            ret_cols=retrieval_cols))
    parser.add_argument('src', nargs=1, help=help_msgs[0])
    parser.add_argument('--enc-cols', '--encoding_cols', dest='enc_cols',
                        default=encoding_cols, required=False,
                        action='append',
                        # nargs='*',
                        help=help_msgs[1])
    parser.add_argument('--ret-cols', '--retrieval_cols', dest='ret_cols',
                        default=retrieval_cols, required=False,
                        action='append',
                        # nargs='*',
                        help=help_msgs[2])
    parser.add_argument('--ons-cols', '--onset_cols', dest='ons_cols',
                        default=onset_cols, required=False, action='append',
                        help=help_msgs[3])
    parser.add_argument('-d', '--dst', dest='dst', nargs='?',
                        help=help_msgs[-1])

    args = parser.parse_args()
    fixit(args.src[0], args.enc_cols, args.ret_cols, args.ons_cols, args.dst)

if __name__ =='__main__':
    main()
