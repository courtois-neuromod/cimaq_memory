#!/usr/bin/python3

import os
import pandas as pd
import re
from argparse import ArgumentParser
from pandas import DataFrame as df
from typing import Union

from load_recursive import load_recursive
from load_cimaq_taskfiles import load_cimaq_taskfiles
from get_default_args import get_default_args
from get_desc import get_desc

revdict = lambda d: dict(tuple((i[1], i[0]) for i in tuple(d.items())))
get_uids = lambda lst, s=0: dict(tuple(enumerate(sorted(list(lst)), s)))

def make_cimaq_stim_map(src:Union[str, os.PathLike],
                       ) -> pd.DataFrame:
    """
    Create a stimuli index for the CIMA-Q project.

    Each stimuli and category is attributed a unique numeric identifier
    consistent across all trials and visits. This helps keep track of
    images and their category. Mostly, this allows homogeneous data types
    (numeric) for participant task files.
    It is recommended by the BIDS Standard. See
    https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html#stimuli
    for details.

    Args:
        src: str, os.PathLike
            The top-level directory of the files, sorted
            according to BIDS Standard.
        categ_pat: str
            String pattern used to identify CIMA-Q stimuli categories.
        ncol: str:
            String representing the column name where and how are the
            CIMA-Q stimuli listed. It is a parameter only in case of
            future change.
        mcol: str
            String representing the column name where and how are the
            CIMA-Q stimuli grouped (old or new - mcol is short for
            map column). It is a parameter only in case of
            future change.

    Returns: pd.DataFrame
        CIMA-Q stimuli indexing table.
    """
    categ_pat = '[a-z]+\-*'
    ncol = 'stim'
    mcol = 'oldnumber'

    categ_pat = re.compile(categ_pat)
    stims = pd.concat([pd.read_csv(itm[-1],
                                   sep='\t', dtype=str,)[['stim', mcol]]
                          for itm in load_cimaq_taskfiles(src)])
    olds = stims.where(stims[mcol].str.contains('old')).dropna(
               axis=0).stim.unique().tolist()
    news = stims.where(stims[mcol].str.contains('new')).dropna(
               axis=0).stim.unique().tolist()

    ustims = df((((itm, categ_pat.findall(itm)[0])
                        if categ_pat.findall(itm)[0] != 'hard'
                        else (itm, categ_pat.findall(itm)[1])
                        for itm in sorted(olds)+sorted(news))),
                      columns=[ncol, 'category'])

    ucateg_ids = get_uids(ustims.category.unique(), 1)
    ustim_ids = get_uids(ustims.stim.unique(), 1)
    ustims['stim_id'] = ustims[ncol].map(revdict(ustim_ids))

    ustims['categ_id'] = ustims['category'].map(revdict(ucateg_ids))
    ustims['old_new'] = ['old' if row[1].stim in olds else 'new'
                               for row in ustims.iterrows()]
    return ustims.set_index('stim_id', drop=True)

def main():
    desc, help_msgs = get_desc(make_cimaq_stim_map)
    parser = ArgumentParser(prog=make_cimaq_stim_map,
                            description=desc,
                            usage=make_cimaq_stim_map.__doc__)
    parser.add_argument('src', nargs=1, help=help_msgs[0])
    parser.add_argument('-c', '--categ-pat', '--categ_pat',
                        dest='categ_pat', nargs='?',
                        help=help_msgs[1])
    parser.add_argument('-n', '--name-col', '--name_col', dest='ncol',
                        default='stim', nargs='?')
    parser.add_argument('-m', '--map-col', '--map_col', dest='mcol',
                        default='oldnumber', nargs='?')
    args = parser.parse_args()
    make_cimaq_stim_map(args.src[0], args.categ_pat, args.ncol, args.mcol)

if __name__ == '__main__':
    main()
