#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import re
import warnings
from glob import glob
from operator import itemgetter
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
    from os.path import basename, dirname
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

def chunks(lst, n):
    """ Yield successive n-sized chunks from lst. """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def img_to_mask3d(source_img:Nifti1Image
                 ) -> Nifti1Image:
    import numpy as np
    from nilearn.image import new_img_like
    return new_img_like(ref_niimg=source_img,
                        data=np.where(source_img.get_fdata() \
                                      > 0, 1, 0))

def format_difumo(indexes:list,
                  target_img:Nifti1Image,
                  dimension:int=256,
                  resolution_mm:int=3,
                  data_dir:Union[str,os.PathLike]=None,
                  **kwargs
                  ) -> Nifti1Image:
    import numpy as np
    from nilearn.datasets import fetch_atlas_difumo
    from nilearn.masking import intersect_masks
    from nilearn import image as nimage
    difumo = fetch_atlas_difumo(dimension=dimension,
                                resolution_mm=resolution_mm,
                                data_dir=data_dir)
    rois = nimage.index_img(nimage.load_img(difumo.maps), indexes)
    rois_resampled = nimage.resample_to_img(source_img=nimage.iter_img(rois),
                                            target_img=target_img,
                                            interpolation='nearest',
                                            copy=True, order='F',
                                            clip=True, fill_value=0,
                                            force_resample=True)
    return intersect_masks(mask_imgs=list(map(img_to_mask3d,
                                              list(nimage.iter_img(rois_resampled)))),
                           threshold=0.0, connected=True)



def trial_fmri(fmri_path:Union[str,os.PathLike, Nifti1Image],
               events_path:Union[str,os.PathLike, pd.DataFrame],
               sep:str='\t', t_r:float=None,
               **kwargs):
    from itertools import starmap
    from more_itertools import flatten
    from nilearn import image as nimage
    import pandas as pd
    # Make pandas Intervals (b:list of beginnigs, e:list of ends)
    mkintrvls = lambda b, e: list(starmap(pd.Interval,tuple(zip(b, e))))
    fmri_img = nimage.load_img(fmri_path)
    if not isinstance(events_path, pd.DataFrame):
        events = pd.read_csv(events_path, sep=sep)
    else:
        events = events_path
    t_r = [t_r if t_r is not None else
           fmri_img.header.get_zooms()[-1]][0]
    frame_times = np.arange(fmri_img.shape[-1]) * t_r
    frame_ends = pd.Series(frame_times).add(t_r).values
    frame_intervals = mkintrvls(pd.Series(frame_times).values,
                                frame_ends)
    trial_ends=(events.onset+abs(events.onset -
                                 events.offset)+events.isi).values
    trial_intervals = mkintrvls(events.onset.values, trial_ends)

    valid_trial_idx = [trial[0] for trial in enumerate(trial_intervals)
                       if trial[1].left<frame_intervals[-1].left]
    valid_trials = pd.Series(trial_intervals).loc[valid_trial_idx].values
#     trial_intervals = list(starmap(pd.Interval,tuple(zip(events.onset.values, trial_ends))))
    bold_by_trial_indx = [[frame[0] for frame in enumerate(frame_intervals)
                           if frame[1].left in trial] for trial in valid_trials]
#     bold_by_trial = list(nimage.index_img(fmri_img, idx)
#                          for idx in bold_by_trial_indx)
    valid_frame_intervals = [pd.Series(frame_intervals).loc[bold_idx].values
                             for bold_idx in bold_by_trial_indx]
    perfo_labels = events.iloc[valid_trial_idx].recognition_performance.fillna('Ctl')
    condition_labels = events.iloc[valid_trial_idx].trial_type
    stim_labels = events.iloc[valid_trial_idx].stim_file.fillna('Ctl').values
    categ_labels = events.iloc[valid_trial_idx].stim_category.fillna('Ctl').values
    
    return pd.DataFrame(tuple(zip(valid_trial_idx,
#                                   bold_by_trial,
                                  bold_by_trial_indx,
                                  valid_trials,
                                  valid_frame_intervals,
                                  condition_labels,
                                  perfo_labels, stim_labels, categ_labels)),
                        columns=['trials',
#                                  'trial_niftis',
                                 'fmri_frames',
                                 'trial_intervals', 'fmri_frame_intervals',
                                 'condition_labels', 'performance_labels',
                                 'stimuli_files', 'category_labels'])

def get_fmri_sessions(topdir:Union[str,os.PathLike],
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
        can be replaced by '*' (Default), equivalent to a UNIX ``find`` pattern.
    """
    # Generate regex and glob patterns
    bold_pattern = '_'.join([f'sub-{sub_id}',f'ses-{ses_id}',f'task-{task}',
                             f'space-{space}',f'desc-{output_type}',
                             f'{modality}{extension}'])
    ev_pattern = f'sub-{sub_id}_ses-{ses_id}_task-{task}_events.tsv'
    # Load fMRI and events files paths into lists
    bold_paths = sorted(map(str, Path(topdir).rglob(f'*{bold_pattern}')))
    event_paths = sorted(map(str,(Path(events_dir).rglob(f'*{ev_pattern}'))))
    # Get only the intersection of these lists
    valid_bold_paths = sorted(boldpath for boldpath in bold_paths
                              if get_sub_ses_key(boldpath) in
                              [get_sub_ses_key(apath) for apath
                               in event_paths])
    valid_event_paths = sorted(evpath for evpath in event_paths
                               if get_sub_ses_key(evpath) in
                               [get_sub_ses_key(apath) for apath
                                in valid_bold_paths])
    # Load corresponding anatomical T1w, brain mask and behavioural file paths
    valid_anat_paths = [get_fmriprep_anat(v_boldpath) for
                        v_boldpath in valid_bold_paths]
    valid_mask_paths = [get_fmriprep_mask(v_boldpath) for
                        v_boldpath in valid_bold_paths]
    valid_behav_paths = [get_behav(v_boldpath, events_dir) for
                         v_boldpath in valid_bold_paths]
    # Zip them together
    zipped_paths = sorted(zip(valid_bold_paths, valid_anat_paths,
                              valid_mask_paths, valid_event_paths,
                              valid_behav_paths))
    # Create sklearn.utils.Bunch objects
    sessions = [Bunch(**dict(zip(['fmri_path', 'anat_path', 'mask_path',
                                  'events_path', 'behav_path', 'sub_id',
                                  'ses_id', 'task', 'space'],
                                 item+Path(item[0]).parts[-4:-2]+(task, space))))
                for item in zipped_paths]
    return sessions

def fetch_fmriprep_session(fmri_path:Union[str,os.PathLike]=None,
                           events_path:Union[str,os.PathLike]=None,
                           strategy:str='Minimal',
                           task:str='memory',
                           space:str='MNI152NLin2009cAsym',
                           anat_mod:str='T1w',
                           session:Union[dict,Bunch]=None,
                           n_sessions:int=1,
                           sub_id:str=None,
                           ses_id:str=None,
                           lc_kws:dict=None,
                           apply_kws:dict=None,
                           clean_kws:dict=None,
                           design_kws:dict=None,
                           glm_kws:dict=None,
                           masker_kws:dict=None,
                           **kwargs):
    import load_confounds
    from inspect import getmembers
    from nilearn import image as nimage
    from sklearn.utils import Bunch
    from pathlib import Path
    
    from cimaq_decoding_params import _params
    from cimaq_decoding_utils import clean_fmri, get_tr, get_frame_times
    events, behav = [pd.read_csv(item, sep='\t') for item in
                     itemgetter(*['events_path', 'behav_path'])(session)]
    # Special contrasts
    events['recognition_performance'] = [row[1].recognition_performance if
                                         row[1].trial_type == 'Enc'
                                         else 'Ctl' for row in events.iterrows()]
    events['ctl_miss_ws_cs'] = ['Cs' if row[1]['recognition_performance'] == 'Hit'
                                else 'Ws' if bool(row[1]['recognition_accuracy'])
                                and not bool(row[1].position_accuracy)
                                else row[1].recognition_performance
                                for row in events.iterrows()]

    loader = dict(getmembers(load_confounds))[f'{strategy}']
    loader = [loader(**lc_kws) if lc_kws is not None
              else loader()][0]
    conf = loader.load(session['fmri_path'])
    fmri_img, mask_img, anat_img = [nimage.load_img(item) for item in
                                    itemgetter(*['fmri_path', 'mask_path',
                                                 'anat_path'])(session)]
    t_r, frame_times = get_tr(fmri_img), get_frame_times(fmri_img)
    _params.design_defs.update(dict(frame_times=frame_times))

    if apply_kws is not None:
        _params.apply_defs.update(apply_kws)    
    if clean_kws is not None:
        _params.clean_defs.update(clean_kws)
    cleaned_fmri = clean_fmri(fmri_img, mask_img, confounds=conf,
                              apply_kws=_params.apply_defs,
                              clean_kws=_params.clean_defs)

    # Argument definitions for each preprocessing step
    target_shape, target_affine = mask_img.shape, cleaned_fmri.affine

    _params.glm_defs.update(dict(t_r=t_r, mask_img=mask_img,
                                 target_shape=target_shape,
                                 target_affine=target_affine))

    if design_kws is not None:
        _params.design_defs.update(design_kws)
    if masker_kws is not None:
        _params.masker_defs.update(masker_kws)
    if glm_kws is not None:
        _params.glm_defs.update(glm_kws)
    
    loaded_attributes = Bunch(events=events, behav=behav,
                              frame_times=frame_times,
                              confounds_loader=loader, confounds=conf,
                              confounds_strategy=strategy,
                              smoothing_fwhm=_params.apply_defs.smoothing_fwhm,
                              fmri_img=fmri_img, mask_img=mask_img,
                              cleaned_fmri=cleaned_fmri)

    default_params = Bunch(clean_defs=_params.clean_defs,
                           design_defs=_params.design_defs,
                           glm_defs=_params.glm_defs,
                           masker_defs=_params.masker_defs)
    session.update(loaded_attributes)
    session.update(default_params)
    return session

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
