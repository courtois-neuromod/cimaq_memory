#!/usr/bin/python3

from argparse import ArgumentParser
from builtins import FutureWarning
import numpy as np
import os
from os import PathLike
import pandas as pd
from pathlib import Path, PosixPath
from tqdm import tqdm as tqdm_
from typing import Union
import warnings

from cimaq_decoding_pipeline import get_fmri_sessions
from cimaq_decoding_pipeline import fetch_fmriprep_session
from cimaq_decoding_pipeline import get_all_contrasts
from cimaq_decoding_utils import flatten
from get_desc import get_desc


def save_signals(fmriprep_dir: Union[str, PathLike, PosixPath],
                 events_dir: Union[str, PathLike, PosixPath],
                 labels_path: Union[str, PathLike,
                                    PosixPath] = None,
                 dst: Union[str, PathLike,
                                 PosixPath] = None,
                 masker_dir: Union[str, PathLike,
                                   PosixPath] = None,
                 trial_type_cols: Union[list, np.array,
                                        pd.Series] = ['trial_type'],
                 **kwargs
                 ) -> None:
    """
    Script to extract signals from all fMRI sessions in CMIA-Q.
    
    See ``cimaq_decoding_pipeline.py`` for implementation details.

    Args:
        fmriprep_dir: str
            Directory to FMRIPrep data.
        events_dir: str
            Directory to CIMA-Q events .tsv files.
        labels_path: str
            Path of predefined feature labels.
        dst: str
            Directory where to save outputs.
        masker_dir: str
            Directory containing pre-fitted, pickled
            ``NiftiMapsMasker`` objects for each session.
        trial_type_cols: array
            Array structure of shape (n_trials, n_conditions).
            Should be removed and placed within ``session`` objects.

    Returns:
        None
    """

    if masker_dir is None:
        masker_dir, labels_path, feature_labels = None, None, None
    else:
        feature_labels = Path(labels_path).read_text().splitlines()

    warnings.filterwarnings(action='ignore', category=FutureWarning)
    
    if dst is None:
        dst = os.path.join(os.getcwd(), 'cimaq_extracted_signals')
    os.makedirs(dst, exist_ok=True)

    if trial_type_cols is None:
        trial_type_cols = ['trial_type']

    defs = dict(output_type='effect_size',
                trial_type_cols=trial_type_cols,
                standardize=True, scale=False,
                maximize=False, feature_labels=feature_labels,
                extract_only=True)

    if kwargs is None:
        kwargs = {}
    defs.update(**kwargs)

    sessions = flatten([get_fmri_sessions(topdir=fmriprep_dir,
                                          events_dir=events_dir,
                              masker_dir=masker_dir, ses_id=ses)
                        for ses in ['V03', 'V10']])

    def ContrastSignalGen(session):
        while True:
            yield get_all_contrasts(session=fetch_fmriprep_session(
                      session=session), **defs)

    [next(ContrastSignalGen(session)).to_csv(os.path.join(dst,
                                                          '_'.join([session.sub_id,
                                                          session.ses_id,
                                                          session.task,
                                                          session.space,
                                                                  'maps-signals.tsv'])),
                                              sep='\t', encoding='UTF-8-SIG',
                                              index='trial_type')
     for session in tqdm_(sessions[94:])]


def main():
    desc, help_msgs = get_desc(save_signals.__doc__)
    parser = ArgumentParser(prog=save_signals,
                            description=desc.splitlines()[0],
                            usage=desc)

    parser.add_argument('fmriprep_dir', nargs=1),#, help=help_msgs[0])
    parser.add_argument('events_dir', nargs=1),#, help=help_msgs[1])
    parser.add_argument('-l', '--labels-path',
                        dest='labels_path', nargs='?')
    parser.add_argument('-d', '--dst', dest='dst', default=None,
                        nargs='?')
    parser.add_argument('-m', '--masker-dir', dest='masker_dir',
                        nargs='?')
    parser.add_argument('-t', '--trial-type-cols',
                          dest='trial_type_cols', nargs='+')

    args = parser.parse_args()
    save_signals(args.fmriprep_dir[0],
                 args.events_dir[0],
                 args.labels_path,
                 args.dst, args.masker_dir)

if __name__ == '__main__':
    main()
