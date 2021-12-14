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
from typing import Union


def get_tr(img:Nifti1Image): return img.header.get_zooms()[-1]

def get_frame_times(img:Nifti1Image): return (np.arange(img.shape[-1]) * get_tr(img))

def get_const_fwhm(img:Nifti1Image): return pow(img.header.get_zooms()[0],2)-1

def get_fmriprep_mask(fmri_path:Union[str,os.PathLike],
                      mask_ext:str='nii.gz',
                      **kwargs):
    bids_patt = lambda p: f'(?<={p})[a-zA-Z0-9]*'
    prefixes = ['sub-','ses-','task-','space-']
    mask_sfx = '_'.join([pf+re.search(bids_patt(pf),
                                      basename(fmri_path)).group()
                         for pf in prefixes]+[f'desc-brain_mask.{mask_ext}'])
    return glob(os.path.join(dirname(fmri_path), mask_sfx))[0]

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

def clean_fmri(fmri_img:Union[str,os.PathLike,Nifti1Image],
               mask_img:Union[str,os.PathLike,Nifti1Image],
               smoothing_fwhm:Union[int,float]=8,
               confounds:pd.DataFrame=None,
               dtype:str='f',
               ensure_finite:bool=False,
               apply_kws:dict=None,
               clean_kws:dict=None) -> Nifti1Image:

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

