#!/usr/bin/env python
# encoding: utf-8
"""
Overview:
Create one map of beta weights per trial using Nistats
https://nistats.github.io/auto_examples/04_low_level_functions/write_events_file.html#sphx-glr-auto-examples-04-low-level-functions-write-events-file-py

1. Load *confounds.tsv file and save as dataframe (to regress out motion, bad frames,
slow drift signal, white matter intensity, etc)

2. Create events dataframe to build design matrices in nistats

3. For each trial, create a design matrix and first-level
https://nistats.github.io/auto_examples/04_low_level_functions/plot_design_matrix.html#sphx-glr-auto-examples-04-low-level-functions-plot-design-matrix-py

4. Fit the model onto data

5. Create contrasts (one per trial) and export corresponding beta maps
"""

import os
import re
import sys

import argparse
import glob
import zipfile

import numpy as np
import pandas as pd
import nilearn
import scipy
import nibabel

from nibabel.nifti1 import Nifti1Image
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from numpy import nan as NaN
from os import PathLike
from pathlib import PosixPath
from sklearn.utils import Bunch
from typing import Union


def preprocess_events(events: Union[str, PathLike, PosixPath,
                                    pd.DataFrame] = None,
                      fmri_img: Union[str, PathLike, PosixPath,
                                      Nifti1Image] = None,
                      session: Union[dict, Bunch] = None
                      ) -> pd.DataFrame:
    if session is not None:
        events, fmri_img = session.events, session.fmri_img
    else:
        fmri_img = nilearn.image.load_img(fmri_img)
    t_r, s_task = fmri_img.header.get_zooms()[-1], events.copy(deep=True)

    scanDur, numCol = fmri_img.shape[-1]*t_r, s_task.shape[1]
    s_task['trial_ends'] = s_task.onset+s_task.duration.values    
    s_task['unscanned'] = (s_task.trial_ends>scanDur).astype(int)
    npad = len(str(s_task.shape[0]))
    s_task['trial_number'] = events.index.astype(str).str.zfill(npad)
    s_task['condition'] = events.trial_number.astype(str)+events.trial_type    
    return s_task[s_task['unscanned']==0]


def sub_tcontrasts1(session:Union[dict,Bunch]=None,
                    sub_id:str=None,
                    tr:float=None,
                    frame_times:list=None,
                    hrf_model:str=None,
                    events:pd.DataFrame=None,
                    fmri_img:Nifti1Image=None,
                    sub_outdir:Union[str,os.PathLike]=None):
    """
    Create beta values maps using nilearn first-level model.

    The beta values correspond to the following contrasts between conditions:
    control, encoding, and encoding_minus_control

    Parameters:
    ----------
    sub_id: string (subject's dccsub_id)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmrsub_idir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """

    if isinstance(session, dict):
        session = Bunch(**session)
    # Model 1: encoding vs control conditions
    events1 = session.events.copy(deep = True)
    cols = ['onset', 'duration', 'trial_type']
    events1 = events1[cols]

    # create the model - Should data be standardized?
    model1 = FirstLevelModel(**session.glm_defs)

    # create the design matrices
    design1 = make_first_level_design_matrix(events=events1, **session.design_defs)

    # fit model with design matrix
    model1 = model1.fit(session.cleaned_fmri, design_matrices = design1)

    # Condition order: control, encoding (alphabetical)
    # contrast 1.1: control condition
    ctl_vec = np.repeat(0, design1.shape[1])
    ctl_vec[0] = 1
    b11_map = model1.compute_contrast(ctl_vec, output_type='effect_size') #"effect_size" for betas
    b11_name = f'betas_{session.sub_id}_ctl.nii'

    #contrast 1.2: encoding condition
    enc_vec = np.repeat(0, design1.shape[1])
    enc_vec[1] = 1
    b12_map = model1.compute_contrast(enc_vec, output_type='effect_size') #"effect_size" for betas
    b12_name = f'betas_{session.sub_id}_enc.nii'

    #contrast 1.3: encoding minus control
    encMinCtl_vec = np.repeat(0, design1.shape[1])
    encMinCtl_vec[1] = 1
    encMinCtl_vec[0] = -1
    b13_map = model1.compute_contrast(encMinCtl_vec, output_type='effect_size') #"effect_size" for betas
    b13_name = f'betas_{session.sub_id}_enc_minus_ctl.nii'
    contrasts = ((b11_map, b11_name), (b12_map, b12_name), (b13_map, b13_name))
    if sub_outdir is not None:
        savedir = os.path.join(sub_outdir, session.sub_id, session.ses_id)
        os.makedirs(savedir, exist_ok=True)
        [nibabel.save(*contrast) for contrast in contrasts]
    return contrasts


def sub_tcontrasts2(session:Union[dict,Bunch]=None,
                    sub_id:str=None,
                    tr:float=None,
                    frame_times:list=None,
                    hrf_model:str=None,
                    events:pd.DataFrame=None,
                    fmri_img:Nifti1Image=None,
                    sub_outdir:Union[str,os.PathLike]=None):
    """
    Create beta values maps using nilearn first-level model.

    The beta values correspond to the following contrasts between conditions:
    hit, miss, hit_minus_miss, hit_minus_ctl and miss_minus_ctl

    Parameters:
    ----------
    sub_id: string (subject's dccsub_id)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmrsub_idir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """
    if isinstance(session, dict):
        session = Bunch(**session)
    # Model 1: encoding vs control conditions
    events2 = session.events.copy(deep = True)
    cols = ['onset', 'duration', 'recognition_performance']
    events2 = events2[cols]
    events2.rename(columns={'recognition_performance':'trial_type'},
                   inplace=True)

    # create the model - Should data be standardized?
    model2 = FirstLevelModel(**session.glm_defs)

    # create the design matrices
    design2 = make_first_level_design_matrix(events=events2,**session.design_defs)

    # fit model with design matrix
    model2 = model2.fit(session.cleaned_fmri, design_matrices = design2)

    # Condition order: control, hit, missed (alphabetical)
    #contrast 2.1: miss
    miss_vec = np.repeat(0, design2.shape[1])
    miss_vec[2] = 1
    b21_map = model2.compute_contrast(miss_vec, output_type='effect_size') #"effect_size" for betas
    b21_name = f'betas_{session.sub_id}_miss.nii'
#     b21_name = os.path.join(sub_outdir, 'betas_sub'+str(sub_id)+'_miss.nii')
#     nibabel.save(b21_map, b21_name)

    #contrast 2.2: hit
    hit_vec = np.repeat(0, design2.shape[1])
    hit_vec[1] = 1
    b22_map = model2.compute_contrast(hit_vec, output_type='effect_size') #"effect_size" for betas
    b22_name = f'betas_{session.sub_id}_hit.nii'

    #contrast 2.3: hit minus miss
    hit_min_miss_vec = np.repeat(0, design2.shape[1])
    hit_min_miss_vec[1] = 1
    hit_min_miss_vec[2] = -1
    b23_map = model2.compute_contrast(hit_min_miss_vec, output_type='effect_size') #"effect_size" for betas
    b23_name = f'betas_{session.sub_id}_hit_minus_miss.nii'

    #contrast 2.4: hit minus control
    hit_min_ctl_vec = np.repeat(0, design2.shape[1])
    hit_min_ctl_vec[1] = 1
    hit_min_ctl_vec[0] = -1
    b24_map = model2.compute_contrast(hit_min_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b24_name = f'betas_{session.sub_id}_hit_minus_ctl.nii'

    #contrast 2.5: miss minus control
    miss_min_ctl_vec = np.repeat(0, design2.shape[1])
    miss_min_ctl_vec[2] = 1
    miss_min_ctl_vec[0] = -1
    b25_map = model2.compute_contrast(miss_min_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b25_name = f'betas_{session.sub_id}_miss_minus_ctl.nii'
    
    contrasts = ((b21_map, b21_name), (b22_map, b22_name), (b23_map, b23_name),
                 (b24_map, b24_name), (b25_map, b25_name))

    if sub_outdir is not None:
        savedir = os.path.join(sub_outdir, session.sub_id, session.ses_id)
        os.makedirs(savedir, exist_ok=True)
        [nibabel.save(*contrast) for contrast in contrasts]
    return contrasts


def sub_tcontrasts3(session:Union[dict,Bunch]=None,
                    sub_id:str=None,
                    tr:float=None,
                    frame_times:list=None,
                    hrf_model:str=None,
                    events:pd.DataFrame=None,
                    fmri_img:Nifti1Image=None,
                    sub_outdir:Union[str,os.PathLike]=None):
    """
    Create beta values maps using nilearn first-level model.

    The beta values correspond to the following contrasts between conditions:
    correctsource (cs), wrongsource (ws), cs_minus_ws, cs_minus_miss,
    ws_minus_miss, cs_minus_ctl, ws_minus_ctl
    hit, miss, hit_minus_miss, hit_minus_ctl and miss_minus_ctl

    Parameters:
    ----------
    sub_id: string (subject's dccsub_id)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmrsub_idir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """
    if isinstance(session, dict):
        session = Bunch(**session)    
    # Model 1: encoding vs control conditions
    events3 = session.events.copy(deep = True)
    cols = ['onset', 'duration', 'ctl_miss_ws_cs']
    events3 = events3[cols]
    events3.rename(columns={'ctl_miss_ws_cs':'trial_type'}, inplace=True)

    # create the model - Should data be standardized?
    model3 = FirstLevelModel(**session.glm_defs)

    # create the design matrices
    design3 = make_first_level_design_matrix(events=events3, **session.design_defs)

    # fit model with design matrix
    model3 = model3.fit(session.cleaned_fmri, design_matrices = design3)

    # Condition order: control, correct source, missed, wrong source (alphabetical)
    #contrast 3.1: wrong source
    ws_vec = np.repeat(0, design3.shape[1])
    ws_vec[3] = 1
    b31_map = model3.compute_contrast(ws_vec, output_type='effect_size') #"effect_size" for betas
    b31_name = f'betas_{session.sub_id}_ws.nii'

    #contrast 3.2: correct source
    cs_vec = np.repeat(0, design3.shape[1])
    cs_vec[1] = 1
    b32_map = model3.compute_contrast(cs_vec, output_type='effect_size') #"effect_size" for betas
    b32_name = f'betas_{session.sub_id}_cs.nii'

    #contrast 3.3: correct source minus wrong source
    cs_minus_ws_vec = np.repeat(0, design3.shape[1])
    cs_minus_ws_vec[1] = 1
    cs_minus_ws_vec[3] = -1
    b33_map = model3.compute_contrast(cs_minus_ws_vec, output_type='effect_size') #"effect_size" for betas
    b33_name = f'betas_{session.sub_id}_cs_minus_ws.nii'

    #contrast 3.4: correct source minus miss
    cs_minus_miss_vec = np.repeat(0, design3.shape[1])
    cs_minus_miss_vec[1] = 1
    cs_minus_miss_vec[2] = -1
    b34_map = model3.compute_contrast(cs_minus_miss_vec, output_type='effect_size') #"effect_size" for betas
    b34_name = f'betas_{session.sub_id}_cs_minus_miss.nii'

    #contrast 3.5: wrong source minus miss
    ws_minus_miss_vec = np.repeat(0, design3.shape[1])
    ws_minus_miss_vec[3] = 1
    ws_minus_miss_vec[2] = -1
    b35_map = model3.compute_contrast(ws_minus_miss_vec, output_type='effect_size') #"effect_size" for betas
    b35_name = f'betas_{session.sub_id}_ws_minus_miss.nii'

    #contrast 3.6: correct source minus control
    cs_minus_ctl_vec = np.repeat(0, design3.shape[1])
    cs_minus_ctl_vec[1] = 1
    cs_minus_ctl_vec[0] = -1
    b36_map = model3.compute_contrast(cs_minus_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b36_name = f'betas_{session.sub_id}_cs_minus_ctl.nii'

    #contrast 3.7: wrong source minus control
    ws_minus_ctl_vec = np.repeat(0, design3.shape[1])
    ws_minus_ctl_vec[3] = 1
    ws_minus_ctl_vec[0] = -1
    b37_map = model3.compute_contrast(ws_minus_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b37_name = f'betas_{session.sub_id}_ws_minus_ctl.nii'

    contrasts = ((b31_map, b31_name), (b32_map, b32_name), (b33_map, b33_name),
                 (b34_map, b34_name), (b35_map, b35_name), (b36_map, b36_name),
                 (b37_map, b37_name))
    if sub_outdir is not None:
        savedir = os.path.join(sub_outdir, session.sub_id, session.ses_id)
        os.makedirs(savedir, exist_ok=True)
        [nibabel.save(*contrast) for contrast in contrasts]

    return contrasts
