#!/usr/bin/python3

# import os
# import pandas as pd
# from pathlib import Path
# from typing import Union

# Not necessary anymore - Use ``load_fmriprep_session`` instead
# def fetch_cimaq(fmriprep_dir:Union[str,os.PathLike],
#                 events_dir:Union[str,os.PathLike],
#                 task:str='memory',
#                 space:str='MNI152NLin2009cAsym',
#                 mdlt:str='T1w',
#                 atlases_dir:Union[str,os.PathLike]=None):

#     from cimaq_utils import absoluteFilePaths, str_inc, seq_eve, seq_odd 

#     anat_suffix = f'*_space-{space}_desc-preproc_{mdlt}.nii.gz'
#     bold_suffix = f'_task-{task}_space-{space}_desc-preproc_bold.nii.gz'
#     mask_suffix = f'_task-{task}_space-{space}_desc-brain_mask.nii.gz'

#     sub_ids = list(next(os.walk(fmriprep_dir)))[1]
#     bolds, masks = [sorted(str_inc([suffix], list(absoluteFilePaths(fmriprep_dir))))
#                     for suffix in [bold_suffix, mask_suffix]]
#     scans = pd.DataFrame(tuple(zip(bolds,masks)),columns=['bolds','masks'])
#     scans[['anats','sub_ids', 'ses_ids']] = \
#         [(next(list(Path(bold).parents)[2].rglob(anat_suffix)),
#           Path(bold).parts[-4],Path(bold).parts[-3]) for bold in bolds]
#     testfiles = tuple(absoluteFilePaths(events_dir))[2:]
#     tests = pd.DataFrame(sorted(zip(seq_eve(testfiles), seq_odd(testfiles))),
#                          columns=['behavs','events'])

#     tests[['sub_ids', 'ses_ids']] = [Path(event).parts[-3:-1]
#                                      for event in tests.events]
#     return pd.merge(scans,tests, on=['sub_ids', 'ses_ids'])