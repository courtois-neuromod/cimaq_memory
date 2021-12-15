#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import re
import warnings
from glob import glob
from nibabel.nifti1 import Nifti1Image
from os.path import basename, dirname
from pathlib import Path
from random import sample
from sklearn.utils import Bunch
from typing import Union

from fetch_fmriprep_session import fetch_fmriprep_session

def get_tr(img:Nifti1Image): return img.header.get_zooms()[-1]

def get_frame_times(img:Nifti1Image): return (np.arange(img.shape[-1]) * get_tr(img))

def get_const_fwhm(img:Nifti1Image): return pow(img.header.get_zooms()[0],2)-1

def get_sub_ses_key(src:Union[str,os.PathLike]
                    ) -> list:
    """
    Return a file's participant and session identifiers in a BIDS-compliant dataset.
    """

    return [p+next(iter(list(filter(None,(re.search(f'(?<={p})[a-zA-Z0-9]*', part)
                                          for part in Path(src).parts))))).group()
            for p in ('sub-', 'ses-')]

def get_fmri_paths(topdir:Union[str,os.PathLike],
                   events_dir:Union[str,os.PathLike]=None,
                   task:str='memory',
                   space:str='MNI152NLin2009cAsym',
                   output_type:str='preproc',
                   extension='.nii.gz',
                   modality='bold',
                   sub_id:str='*',
                   ses_id:str='*',
                   **kwargs
                   ) -> list:

    """
    Return a sorted list of the desired BOLD fMRI nifti file paths.
    
    Args:
        topdir: str or os.PathLike
            Database top-level directory path.

        events_dir: str or os.PathLike
            Directory where the events and/or behavioural files are stored.
            If None is provided, it is assumed to be identical as ``topdir``.
        
        task: str (Default = 'memory')
            Name of the experimental task (i.e. 'rest' is also valid).

        space: str (Default = 'MNI152NLin2009cAsym')
            Name of the template used during resampling.
            Most likely corresponding to a valid TemplateFlow name.
        
        output_type: str (Default = 'preproc')
            Name of the desired FMRIPrep output type.
            Most likely corresponding to a valid FMRIPrep output type name.
            
        extension: str (Default = '.nii.gz')
            Nifti files extension. The leading '.' is required.
        
        modality: str (Default = 'bold')
            Scanning modality used during the experiment.
        
        sub_id: str (Default = '*')
            Participant identifier. By default, returns all participants.
            The leading 'sub-' must be omitted.
            If the identifier is numeric, it should be quoted.

        ses_id: str (Default = '*')
            Session identifier. By default, returns all sessions.
            If the identifier is numeric, it should be quoted.
            The leading 'ses-' must be omitted.
        
    Returns: list
        Sorted list of the desired nifti file paths.
    
    Notes:
        All parameters excepting ``topdir`` and ``events_dir``
        can be replaced by '*', equivalent to a UNIX ``find`` pattern.
    """

    bold_pattern = '_'.join([f'sub-{sub_id}',f'ses-{ses_id}',f'task-{task}',
                             f'space-{space}',f'desc-{output_type}',
                             f'{modality}{extension}'])
    ev_pattern = f'sub-{sub_id}_ses-{ses_id}_task-{task}_events.tsv'
    bold_paths = sorted(map(str, Path(topdir).rglob(f'*{bold_pattern}')))
    
    event_paths = sorted(map(str,(Path(events_dir).rglob(f'*{ev_pattern}'))))
    return sorted(boldpath for boldpath in bold_paths if get_sub_ses_key(boldpath)
                  in [get_sub_ses_key(apath) for apath in event_paths])


def fetch_cimaq(topdir:Union[str,os.PathLike],
                events_dir:Union[str,os.PathLike],
                at_random:bool=True,
                n_ses:int=1,
                search_kws:dict=None,
                **kwargs) -> Bunch:
    if search_kws is not None:
        fmri_paths = get_fmri_paths(topdir,events_dir,
                                    **search_kws)
    else:
        fmri_paths = get_fmri_paths(topdir,events_dir)
    if at_random is True:
        fmri_paths = sample(fmri_paths, n_ses)
        
    return Bunch(data=(fetch_fmriprep_session(apath, events_dir)
                       for apath in fmri_paths))


# FMRIPrep outputs path matching functions
def get_fmriprep_anat(fmri_path, mdlt:str='T1w',
                      ext:str='nii.gz',
                      **kwargs):
    space = re.search(f'(?<=_space-)[a-zA-Z0-9]*',
                      basename(fmri_path)).group()
    anat_suffix = f'*_space-{space}_desc-preproc_{mdlt}.{ext}'
    return str(next(list(Path(fmri_path).parents)[2].rglob(anat_suffix)))


def get_events(fmri_path, events_dir)->str:
    sub_id, ses_id = Path(fmri_path).parts[-4:-2]
    globbed = glob(os.path.join(events_dir,
                                *Path(fmri_path).parts[-4:-2],
                                '*events.tsv'))
    return [False if globbed == [] else globbed[0]][0]


def get_behav(fmri_path, events_dir)->str:
    sub_id, ses_id = Path(fmri_path).parts[-4:-2]
    globbed = glob(os.path.join(events_dir,
                                *Path(fmri_path).parts[-4:-2],
                                '*behavioural.tsv'))
    return [False if globbed == [] else globbed[0]][0]


def get_fmriprep_mask(fmri_path:Union[str,os.PathLike],
                      mask_ext:str='nii.gz',
                      **kwargs):
    bids_patt = lambda p: f'(?<={p})[a-zA-Z0-9]*'
    prefixes = ['sub-','ses-','task-','space-']
    mask_sfx = '_'.join([pf+re.search(bids_patt(pf),
                                      basename(fmri_path)).group()
                         for pf in prefixes]+[f'desc-brain_mask.{mask_ext}'])
    return glob(os.path.join(dirname(fmri_path), mask_sfx))[0]


# FMRIPrep BOLD-to-mask image alignment and signal cleaning
def clean_fmri(fmri_img:Union[str,os.PathLike,Nifti1Image],
               mask_img:Union[str,os.PathLike,Nifti1Image],
               smoothing_fwhm:Union[int,float]=8,
               confounds:pd.DataFrame=None,
               dtype:str='f',
               ensure_finite:bool=False,
               apply_kws:dict=None,
               clean_kws:dict=None,
               **kwargs) -> Nifti1Image:

    warnings.simplefilter('ignore', category=FutureWarning)
    from nilearn import image as nimage
    from nilearn.masking import apply_mask, unmask
    from nilearn.signal import clean
    from cimaq_decoding_params import _params
    
    apply_defs = _params.apply_defs
    clean_defs = _params.clean_defs
    if apply_kws is not None:
        apply_defs.update(apply_kws)
    if clean_kws is not None:
        clean_defs.update(clean_kws)
    fmri_img = nimage.load_img(fmri_img)
    mask_img = nimage.load_img(mask_img)
    cleaned_fmri = unmask(clean(apply_mask(fmri_img, mask_img,
                                           **apply_defs),
                                confounds=confounds,
                                **clean_defs),
                          mask_img)
    cleaned_fmri = nimage.new_img_like(fmri_img, cleaned_fmri.get_fdata(),
                                       copy_header=True)
    return cleaned_fmri

