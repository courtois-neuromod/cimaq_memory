#!/usr/bin/python3

import nilearn
import numpy as np
import os
import pandas as pd
import pathlib
import re
import tqdm
import typing
import warnings

from collections import defaultdict
from glob import glob
from io import StringIO
from nibabel.nifti1 import Nifti1Image
from nilearn import image as nimage
from nilearn import plotting as niplot
from nilearn.datasets import fetch_atlas_difumo
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.input_data import MultiNiftiMasker, NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker, NiftiMasker
from nilearn.input_data import NiftiSpheresMasker
from operator import itemgetter
from os import PathLike
from os.path import basename, dirname
from pathlib import Path, PosixPath
from random import sample
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.utils import Bunch
from tqdm import tqdm as tqdm_
from typing import Iterable, Sequence, Union


########################################################################
# Utility Functions & Snippets
########################################################################

def get_t_r(img: Nifti1Image):
    """
    Return a ``Nifti1Image`` scan repetition time from its header.
    """

    return img.header.get_zooms()[-1]


def get_frame_times(img: Nifti1Image):
    """
    Return scan frame onset times based on the repetition time of ``img``.
    """

    return (np.arange(img.shape[-1]) * get_t_r(img))


def get_const_fwhm(img: Nifti1Image):
    """
    Return the square of a ``Nifti1Image`` voxel width minus 1 as a float.
    """

    return pow(img.header.get_zooms()[0], 2)-1


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten(nested_seq: Union[Iterable, Sequence]) -> list:
    """
    Return vectorized (1D) list from nested Sequence ``nested_seq``.
    """

    return [bottomElem for sublist in nested_seq for bottomElem
            in (flatten(sublist)
                if (isinstance(sublist, Sequence)
                    and not isinstance(sublist, str))
                else [sublist])]


def ig_f(src: Union[str, PathLike, PosixPath]
         ) -> list:
    """
    Returns only file paths within a directory.

    Useful to pass as an the ``ignore`` parameter from
    ``shutil.copytree``. Allows to recursively copy a
    directory tree without the files.
    """

    return sorted(filter(os.path.isfile, sorted(Path(src).rglob('*'))))


def factorGenerator(n: int) -> typing.Generator:
    """
    Return a generator an integer's factors.
    """

    from functools import reduce
    yield from sorted(set(reduce(list.__add__,
                                 ([i, n//i] for i in range(1, int(n**0.5) + 1)
                                  if n % i == 0))))[1:-1].__iter__()


########################################################################
# FMRIPrep outputs path matching functions
########################################################################


def get_sub_ses_key(fmri_path: Union[str, PathLike, PosixPath]
                    ) -> list:
    """
    Return a participant and session identifiers in a BIDS-compliant dataset.
    """
    prfs = ('sub-', 'ses-')
    patterns = [f'(?<={prf})[a-zA-Z0-9]*'
                for prf in prfs]

    sub_id, ses_id = [re.search(pat, fmri_path).group()
                      for pat in patterns]

    return (prfs[0]+sub_id, prfs[1]+ses_id)


def get_fmriprep_anat(fmri_path: Union[str, PathLike, PosixPath],
                      mdlt: str = 'T1w',
                      ext: str = 'nii.gz',
                      **kwargs):
    space = re.search(f'(?<=_space-)[a-zA-Z0-9]*',
                      basename(fmri_path)).group()
    anat_suffix = f'*_space-{space}_desc-preproc_{mdlt}.{ext}'
    return str(next(list(Path(fmri_path).parents)[2].rglob(anat_suffix)))


def get_events(fmri_path: Union[str, PathLike, PosixPath],
               events_dir: Union[str, PathLike, PosixPath]
               ) -> str:
    sub_id, ses_id = Path(fmri_path).parts[-4:-2]
    globbed = glob(os.path.join(events_dir,
                                *Path(fmri_path).parts[-4:-2],
                                '*events.tsv'))
    return [False if globbed == [] else globbed[0]][0]


def get_behav(fmri_path: Union[str, PathLike, PosixPath],
              events_dir: Union[str, PathLike, PosixPath]
              ) -> str:
    sub_id, ses_id = Path(fmri_path).parts[-4:-2]
    globbed = glob(os.path.join(events_dir,
                                *Path(fmri_path).parts[-4:-2],
                                '*behavioural.tsv'))
    return [False if globbed == [] else str(globbed[0])][0]


def get_fmriprep_mask(fmri_path: Union[str, PathLike, PosixPath],
                      mask_ext: str = 'nii.gz',
                      **kwargs):
    from os.path import basename, dirname
    def bids_patt(p): return f'(?<={p})[a-zA-Z0-9]*'
    prefixes = ['sub-', 'ses-', 'task-', 'space-']
    mask_sfx = '_'.join([pf+re.search(bids_patt(pf),
                                      basename(fmri_path)).group()
                         for pf in prefixes]+[f'desc-brain_mask.{mask_ext}'])
    return glob(os.path.join(dirname(fmri_path), mask_sfx))[0]

def get_maps_masker_path(fmri_path: Union[str, PathLike, PosixPath],
                    masker_dir: Union[str, PathLike, PosixPath],
                    dimension: int = 693,
                    resolution_mm: int = 3
                    ) -> NiftiMapsMasker:

    prefix = '_'.join(os.path.basename(session.fmri_path).split('_')[:-2])
    masker_path = sorted(Path(masker_dir).rglob(f'{prefix}*.pickle'))
    if masker_path == []:
        masker_path = None,
    else:
        masker_path = masker_path[0]
    return masker_path


def unpickle(src, encoding: str = 'UTF-8',
             **kwargs
             ):
    with open(src, mode='rb') as pickled:
        wanted = pickle.load(pickled, encoding=encoding,
                             **kwargs)
    mfile.close()
    return wanted


def save_masker(dst: Union[str, PathLike, PosixPath],
                masker,
                sub_id: str = None,
                ses_id: str = None,
                task: str = None,
                space: str = None,
                session: Union[dict, Bunch] = None,
                masker_name: str = 'maps-masker.pickle',
                protocol: int = -1,
                **kwargs
                ) -> None:

    import pickle
    from operator import itemgetter

    if session is not None:
        attrs = ['sub_id', 'ses_id', 'task', 'space']
        sub_id, ses_id, task, space = itemgetter(*attrs)(session)
    sub_dst = os.path.join(dst, sub_id, ses_id)
    masker_str = '_'.join([sub_id, ses_id, f'task-{task}',
                              f'space-{space}'])
    os.makedirs(dst, exist_ok=True)
    os.makedirs(sub_dst, exist_ok=True)
    dims = f'{str(int(masker.maps_img.shape[-1]))}'
    resol = f'{str(int(masker.maps_img.header.get_zooms()[0]))}mm'
    masker_name = f'cortex-difumo-{dims}-{resol}-{masker_name}'

    masker_path = os.path.join(sub_dst,
                               '_'.join([masker_str,
                                         masker_name]))

    with open(masker_path, mode='wb') as mfile:
        pickle.dump(obj=masker, file=mfile, protocol=protocol)
        mfile.close()
