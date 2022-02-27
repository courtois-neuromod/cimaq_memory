#!/usr/bin/python3

import nilearn
import numpy as np
import os
import pandas as pd
import pathlib
import re
import warnings

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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.utils import Bunch
from tqdm import tqdm
from typing import Iterable, Sequence, Union


########################################################################
# Utility Functions & Snippets
########################################################################


def get_t_r(img:Nifti1Image):
    return img.header.get_zooms()[-1]

def get_frame_times(img:Nifti1Image):
    return (np.arange(img.shape[-1]) * get_t_r(img))

def get_const_fwhm(img:Nifti1Image):
    return pow(img.header.get_zooms()[0],2)-1

def get_sub_ses_key(src:Union[str,os.PathLike]
                    ) -> list:
    """
    Return a file's participant and session identifiers in a BIDS-compliant dataset.
    """

    return [p+next(iter(list(filter(None,(re.search(f'(?<={p})[a-zA-Z0-9]*', part)
                                          for part in Path(src).parts))))).group()
            for p in ('sub-', 'ses-')]

def chunks(lst, n):
    """ Yield successive n-sized chunks from lst. """
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


def get_difumo(dimension: int,
               resolution_mm: int,
               data_dir: Union[str, PathLike,
                               PosixPath] = None
              ) -> Bunch:

    """
    Args:
        dimension: int
            Desired number of ROIs in the map.
            Valid choices are 64, 128, 256, 512 and 1024.
            Note that runtime increases proportionally with
            the number of dimensions.

        resolution_mm: int
            Desired voxel size in mm
            Valid choices are 2 or 3

        data_dir: str, os.PathLike or pathlib.PosixPath (Default=None)
            Directory where the atlases are located.
            If ``None`` (default) is provided, the current
            working directory is used.
            If the requested atlas is not present, it will be downloaded with
            ``nilearn.datasets.fetch_atlas_difumo``.

    Returns: Bunch
        Dict-like mapping with 'maps' and 'labels' as keys.
        The keys's respective values types are
        ``nibabel.nifti1.Nifti1Image`` (4D) and
        ``pandas.DataFrame`` (index=range(1, ``dimensions``, name='components').
    """

    suffix = f'*{dimension}/{resolution_mm}mm/*.nii.gz'
    if data_dir is None:
        data_dir = os.getcwd()
    maps_path = sorted(Path(data_dir).rglob(suffix))
    if maps_path == []:
        fetch_atlas_difumo(dimension, resolution_mm,
                           data_dir, resume=True)

    maps = list(Path(data_dir).rglob(f'*{dimension}/{resolution_mm}mm/*.nii.gz'))[0]
    labels = list(Path(data_dir).rglob(f'*{dimension}/*.csv'))[0]
    maps, labels = tuple(map(str, (maps, labels)))
    labels_buff = StringIO(Path(labels).read_bytes().decode('UTF-8').lower())
        
    maps = nimage.load_img(maps)
    labels = pd.read_csv(labels_buff).set_index('component')
    return Bunch(maps=maps, labels=labels)


def get_difumo_cut_coords(dimension: int,
                          resolution_mm: int,
                          data_dir: Union[str, PathLike,
                                          PosixPath] = None,
                          output_dir: Union[str, PathLike,
                                            PosixPath] = None,
                          as_dataframe: bool = True
                          ) -> None:
    """
    Find each atlas ROI's in a DiFuMo atlas (4th dimension) MNI coordinates.
    
    Args:
        dimension: int
            Desired number of ROIs in the map.
            Valid choices are 64, 128, 256, 512 and 1024.
            Note that runtime increases proportionally with
            the number of dimensions.

        resolution_mm: int
            Desired voxel size in mm.
            Valid choices are 2 or 3.

        data_dir: str or os.PathLike
            Directory where the atlases are located.
            If ``None`` (default) is provided, the current
            working directory is used.            
            If the requested atlas is not present, it will be downloaded with
            ``nilearn.datasets.fetch_atlas_difumo``.

        output_dir: str or os.PathLike (Default = None)
            Directory where to save the coordinates DataFrame as a tsv file.
            If ``None`` (default) is provided, saves in the current
            working directory.
        
        as_dataframe: bool (Default=True)
            When True (default), returns the constructed DataFrame.
            Returns ``None`` otherwise.
    
    Returns: pd.DataFrame or None
        Return value depends on the ``as_dataframe`` parameter.
    """
    
    if data_dir is None:
        data_dir = os.getcwd()    

    # Load DiFuMo Atlas map as Nifti image & labels as pandas DataFrame
    difumo = get_difumo(data_dir=data_dir, dimension=dimension,
                        resolution_mm=resolution_mm)

    # Find Each Atlas ROI's MNI Coordinates
    difumo_cut_coords = niplot.find_probabilistic_atlas_cut_coords(nimage.load_img(difumo.maps))

    # Use the Coordinates as Seeds for the NiftiSpheresMasker
    spheres_masker = NiftiSpheresMasker(seeds=difumo_cut_coords,standardize=True)

    # Save the Coordinates and ROI Labels to csv
    # Coordinates should be equivalent respective to each atlas map images
    difumo_coords = pd.DataFrame(difumo_cut_coords, columns=['x','y','z'],
                                 index=difumo.labels.difumo_names).reset_index()
    difumo_coords.set_axis(range(1,difumo.labels.shape[0]+1), axis=0, inplace=True)
    difumo_coords.reset_index(drop=False, inplace=True)
    difumo_coords.set_axis(['component', 'difumo_names', 'x', 'y', 'z'],
                               axis=1, inplace=True)
    difumo_coords = difumo_coords.set_index('component')
    
    if output_dir is not None:
        fname_base = f'difumo_{dimension}_dims_{str(resolution_mm)}mm'
        savename = '_'.join([fname_base, 'cut_coords.tsv'])
        difumo_coords.to_csv(os.path.join(output_dir, savename),
                             sep='\t', index='component', encoding='UTF-8-SIG')
    if as_dataframe is True:
        return difumo_coords


########################################################################        
# Trial-based contrast definition and weighting
########################################################################


def get_glm_events(events: Union[str, PathLike, PosixPath,
                                 pd.DataFrame],
                   trial_type_cols: list = None,
                   **kwargs):
    """
    Return nilearn compatible events for each syncrhonous contrasts.

    Changing 'trial_type' to 'index' to get unique trials,
    a different design matrix can be obtained for each
    computation of interest.

    Allows OneVsOneClassification.
    """
    from operator import itemgetter
    ntrials, npad = events.shape[0], len(str(events.shape[0]))
    trials_ = events.reset_index(drop=False).index.astype(str).str.zfill(npad).tolist()

    if trial_type_cols is None:
        trial_type_cols = ['trial_type']

    trial_labels = [trials_]+[events[col].tolist() for col in trial_type_cols]
    return [pd.DataFrame(zip(events.onset, events.duration, labels),
                         columns=['onset','duration','trial_type'])
            for labels in trial_labels]


def manage_duplicates(X, method='mean', axis=0):
    """
    Returns ``X`` without duplicate labels along ``axis``.
    """

    from inspect import getmembers
    dups = X.loc[:,X.columns.value_counts()>1].columns.unique()
    if len(dups) == 0:
        return X
    mthd = dict(getmembers(pd.DataFrame))[f'{method}']
    newdata = [pd.Series(data=X[dup].T.apply(mthd), name=dup)
               for dup in dups]
    return pd.concat([X.copy(deep=True).drop(dups, axis=1),
                      pd.concat(newdata, axis=1)], axis=1)


def weight_signals(signals, weights, labels,
                   method='corrcoef',
                   labelize=True,
                   keep_zero_var=False):
    """
    Returns the product of the contrasts obtained with ``get_glm_events``.
    
    Signals, weights and labels should be aligned along a common axis.
    Otherwise, ValueError is raised.
    """
    
    from inspect import getmembers
    from sklearn.feature_selection import VarianceThreshold
    signals = signals.set_index(labels)
    mthd = dict(getmembers(np))[f'{method}']
    if all([dim not in weights.shape for dim in signals.shape]):
        weights = weights.apply(mthd)
    for weight in weights.index:
        weights.set_axis(signals.columns, axis=1, inplace=True)
        signals.loc[weight] = signals.loc[weight]*weights.loc[weight]
    if keep_zero_var is False:
        cols = VarianceThreshold().fit(signals).get_support(indices=True)
        signals = signals.iloc[:, cols]
    if labelize is True:
        signals = signals.set_index(labels)
    return signals, labels


# def weight_signals(signals, weights, labels, keep_zero_var=False):
#     """
#     Returns the product of the contrasts obtained with ``get_glm_events``.
#     """

#     from sklearn.feature_selection import VarianceThreshold
#     signals = signals.set_index(labels)
#     for weight in weights.index:
#         signals.loc[weight] = signals.loc[weight]*weights.loc[weight]
#     if keep_zero_var is False:
#         cols = VarianceThreshold().fit(signals).get_support(indices=True)
#         signals = signals.iloc[:, cols]
#     return signals, labels


########################################################################        
# FMRIPrep outputs path matching functions
########################################################################


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


########################################################################
# FMRIPrep BOLD-to-mask image alignment and signal cleaning
########################################################################


def clean_scale_mask_img(fmri_img: Union[str, PathLike,
                                         PosixPath, Nifti1Image],
                         mask_img: Union[str, PathLike,
                                         PosixPath, Nifti1Image],
                         scaler: [MinMaxScaler, MaxAbsScaler, bool,
                                  Normalizer, OneHotEncoder] = None,
                         confounds: Union[str, PathLike, PosixPath,
                                          pd.DataFrame] = None,
                         t_r: float = None,
                         **kwargs):

    from nilearn.image import load_img, new_img_like
    from nilearn.masking import apply_mask, unmask
    from nilearn.signal import clean

    fmri_img, mask_img = tuple(map(load_img, (fmri_img, mask_img)))

    if t_r is None:
        t_r = fmri_img.header.get_zooms()[-1]

    signals = apply_mask(imgs=fmri_img, dtype='f',
                         mask_img=mask_img)

    defaults = Bunch(runs=None, filter=None,
                     low_pass=None, high_pass=None,
                     detrend=False, standardize=False,
                     standardize_confounds=False,
                     ensure_finite=True)

    if kwargs is None:
        kwargs = {}
    defaults.update(kwargs)

    signals = clean(signals=signals, t_r=t_r,
                    confounds=confounds,
                    **defaults)

    if scaler is not False:
        if scaler is None:
            scaler = StandardScaler()
        signals = scaler.fit(signals).transform(signals)

    new_img = unmask(signals, mask_img=mask_img)
    return new_img_like(fmri_img,
                        new_img.get_fdata(),
                        copy_header=True)


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


########################################################################
# Main Pipeline (Initial Steps)
########################################################################


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


def get_contrasts(fmri_img=None,
                  events=None,
                  design_kws: Union[dict, Bunch] = None,
                  glm_kws: Union[dict, Bunch] = None,
                  trial_type_cols: list = None,
                  method='mean',
                  output_type: str = 'effect_variance',
                  masker: [MultiNiftiMasker, NiftiLabelsMasker,
                           NiftiMapsMasker, NiftiMasker] = None,
                  labels: Sequence = None,
                  session=None,
                     **kwargs):

    design_defs, glm_defs = {}, {}

    if session is not None:
        fmri_img, events = itemgetter(*['cleaned_fmri',
                                        'events'])(session)
        design_defs.update(session.design_defs)
        glm_defs.update(session.glm_defs)

    t_r = get_t_r(fmri_img)
    frame_times = get_frame_times(fmri_img)

    if design_kws is not None:
        design_defs.update(design_kws)
    if glm_kws is not None:
        glm_defs.update(glm_kws)

    design = make_first_level_design_matrix(frame_times, events=events,
                                            drift_model=None,
                                            **design_defs)

    model = FirstLevelModel(**glm_defs).fit(run_imgs=fmri_img,
                                            design_matrices=design)
    contrasts = nimage.concat_imgs([model.compute_contrast(trial,
                                                           output_type=output_type)
                                    for trial in tqdm(design.columns.astype(str),
                                                      desc='Computing Contrasts')])
    if masker is None:
        masker = NiftiMasker().fit(contrasts)

    signals = masker.transform_single_imgs(contrasts)
    signals = manage_duplicates(pd.DataFrame(signals, columns=labels,
                                             index=design.columns).iloc[:-1, :],
                                method=method)
    return Bunch(model=model, contrast_img=contrasts, signals=signals)


def preprocess_events(events: Union[str, PathLike, PosixPath,
                                    pd.DataFrame] = None,
                      fmri_img: Union[str, PathLike, PosixPath,
                                      Nifti1Image] = None,
                      session: Union[dict, Bunch] = None
                      ) -> pd.DataFrame:
    """
    CIMA-Q-Specific Event-File Correction.
    
    The ``duration`` column only shows stimulus presentation
    duration, which is always 3s. However, each trial is not
    3s-long. The ISI must be accounted for.
    Some experimental trials were still ongoing after the
    scanner stopped recording brain signals.
    These trials must be left out.
    Retrieval task performance must be synchronized
    on a stimuli-trial basis with the encoding task.
    
    Args:
        events: str, PathLike, PosixPath or pd.DataFrame (Default=None).
            Path to a '.tsv' file or an in-memory pd.DataFrame.
            Experimental paradigm description data.

        fmri_img: str, PathLike, PosixPath or Nifti1Image (Default=None).
            Path to a '.nii' or '.nii.gz' file or an
            in-memory nibabel.Nifti1Image 4D image.
            fMRI BOLD signal for the same session as ``events``.
        
        session: dict or Bunch (default=None)
            Dict or ``sklearn.utils.Bunch`` minimally containg
            all of the above parameters just like keyword arguments.
    
    Returns: pd.DataFrame
        Corrected events.
    """

    if session is not None:
        events, fmri_img = session.events, session.fmri_img
    else:
        fmri_img = nilearn.image.load_img(fmri_img)
        events = [events if isinstance(events, pd.DataFrame)
                  else pd.read_csv(events, sep='\t')][0]
    t_r, s_task = fmri_img.header.get_zooms()[-1], events.copy(deep=True)
    s_task['duration'] = s_task['duration']+s_task['isi']

    scanDur, numCol = fmri_img.shape[-1]*t_r, s_task.shape[1]
    s_task['trial_ends'] = s_task.onset+s_task.duration.values    
    s_task['unscanned'] = (s_task.trial_ends>scanDur).astype(int)
    s_task = s_task[s_task['unscanned']==0]
    # Special contrasts
    s_task['recognition_performance'] = [row[1].recognition_performance if
                                         row[1].trial_type == 'Enc'
                                         else 'Ctl' for row in s_task.iterrows()]
    s_task['ctl_miss_ws_cs'] = ['Cs' if row[1]['recognition_performance'] == 'Hit'
                                else 'Ws' if bool(row[1]['recognition_accuracy'])
                                and not bool(row[1].position_accuracy)
                                else row[1].recognition_performance
                                for row in s_task.iterrows()]
    s_task.duration = s_task.duration+s_task.isi
    return s_task


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
    from cimaq_decoding_utils import clean_fmri, get_t_r, get_frame_times
    events, behav = [pd.read_csv(item, sep='\t') for item in
                     itemgetter(*['events_path', 'behav_path'])(session)]

    loader = dict(getmembers(load_confounds))[f'{strategy}']
    loader = [loader(**lc_kws) if lc_kws is not None
              else loader()][0]
    conf = pd.DataFrame(loader.load(session['fmri_path']))
    fmri_img, mask_img, anat_img = [nimage.load_img(item) for item in
                                    itemgetter(*['fmri_path', 'mask_path',
                                                 'anat_path'])(session)]
    t_r, frame_times = get_t_r(fmri_img), get_frame_times(fmri_img)
    events = preprocess_events(events, fmri_img)
#    _params.design_defs.update(dict(frame_times=frame_times))

    if apply_kws is not None:
        _params.apply_defs.update(apply_kws)    
    if clean_kws is not None:
        _params.clean_defs.update(clean_kws)
    cleaned_fmri = clean_fmri(fmri_img, mask_img,
                              confounds=conf,
                              apply_kws=_params.apply_defs,
                              clean_kws=_params.clean_defs)

    # Argument definitions for each preprocessing step
    target_shape, target_affine = mask_img.shape, cleaned_fmri.affine

    _params.glm_defs.update(Bunch(target_shape=target_shape,
                                  target_affine=target_affine,
                                  subject_label='_'.join([session.sub_id,
                                                          session.ses_id])))
    _params.design_defs.update(Bunch(frame_times=frame_times))

    if design_kws is not None:
        _params.design_defs.update(design_kws)
    if masker_kws is not None:
        _params.masker_defs.update(masker_kws)
    if glm_kws is not None:
        _params.glm_defs.update(glm_kws)
    
    loaded_attributes = Bunch(events=events, behav=behav,
                              frame_times=frame_times,
                              t_r=t_r,
                              confounds_loader=loader, confounds=conf,
                              confounds_strategy=strategy,
                              smoothing_fwhm=_params.apply_defs.smoothing_fwhm,
                              anat_img=anat_img, fmri_img=fmri_img, mask_img=mask_img,
                              cleaned_fmri=cleaned_fmri)

    default_params = Bunch(apply_defs=_params.apply_defs,
                           clean_defs=_params.clean_defs,
                           design_defs=_params.design_defs,
                           glm_defs=_params.glm_defs,
                           masker_defs=_params.masker_defs)
    session.update(loaded_attributes)
    session.update(default_params)
    session.get_t_r, session.get_frame_times = get_t_r, get_frame_times
    return session


def get_iso_labels(frame_times:Iterable=None,
                   t_r:float=None,
                   events:pd.DataFrame=None,
                   session:Union[dict, Bunch]=None,
                   **kwargs) -> list:
    """
    Return a list of which trial condition a given fMRI frame fits into.
    
    Args:
        frame_times: Iterable (default=None)
            Iterable array containing the onset of each fMRI frame
            since scan start.

        t_r: float (default=None)
            fMRI scan repetition time, in seconds.

        events: pd.DataFrame (default=None)
            DataFrame containing experimental procedure details.
            The required columns are the same as used by Nilearn functions
            (e.g. ``nilearn.glm.first_level.FirstLevelModel``).
            Those are ["onset", "duration", "trial_type"].
            Other columns are ignored.

        session: dict or Bunch (default=None)
            Dict or ``sklearn.utils.Bunch`` minimally containg
            all of the above parameters just like keyword arguments.

    Returns: list
        List of which trial condition a given fMRI frame fits into.
        The list is of the same lenght as ``frame_times``,
        suitable for classification and statistical operations.
    """

    if session is not None:
        params = itemgetter(*('frame_times','t_r','events'))(session)
        frame_times,t_r,events = params

    frame_intervals = [pd.Interval(*item) for item in
                       tuple(zip(frame_times, frame_times+t_r))]
    trial_ends = (events.onset+events.duration).values
    trial_intervals = [pd.Interval(*item) for item in
                       tuple(zip(events.onset.values, trial_ends))]
    bold_by_trial_indx = [[frame[0] for frame in
                           enumerate(frame_intervals)
                           if frame[1].left in trial]
                          for trial in trial_intervals]
    labels = [events.trial_type.tolist()[0]]
    [labels.extend(lst) for lst in
     [[item[0]]*len(item[1]) for item in
      tuple(zip(events.trial_type.values,
                bold_by_trial_indx))][1:]]
    return np.array(labels)


########################################################################
# Unused yet
########################################################################
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
