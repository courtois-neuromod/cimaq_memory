#!/usr/bin/python3

import nilearn
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import typing

from nibabel.nifti1 import Nifti1Image
from nilearn import image as nimage
from nilearn.input_data import MultiNiftiMasker, NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker, NiftiMasker
from operator import itemgetter
from os import PathLike
from os.path import splitext
from pathlib import Path, PosixPath
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from tqdm import tqdm as tqdm_
from typing import Iterable, Sequence, Union

from cimaq_decoding_utils import flatten, get_behav
from cimaq_decoding_utils import get_events, unpickle
from cimaq_decoding_utils import get_fmriprep_anat, get_fmriprep_mask
from cimaq_decoding_utils import get_frame_times, get_sub_ses_key, get_t_r


########################################################################
# Main Pipeline: Initial Steps and Data Fetching
########################################################################


def get_fmri_sessions(topdir: Union[str, PathLike, PosixPath],
                      events_dir: Union[str, PathLike,
                                        PosixPath] = None,
                      masker_dir: Union[str, PathLike,
                                        PosixPath] = None,
                      task: str = 'memory',
                      space: str = 'MNI152NLin2009cAsym',
                      output_type: str = 'preproc',
                      extension: str = '.nii.gz',
                      modality: str = 'bold',
                      sub_id: str = '*',
                      ses_id: str = '*',
                      **kwargs
                      ) -> list:
    """
    Return a sorted list of the desired BOLD fMRI nifti file paths.

    Args:
        topdir: str, PathLike, or PosixPath (Default = None)
            Database top-level directory path.

        events_dir: str, PathLike, or PosixPath (Default = None)
            Directory where the events and/or behavioural
            files are stored. If None is provided, it is assumed
            to be identical as ``topdir``.

        masker_dir: str, PathLike, or PosixPath (Default = None)
            Directory where prefitted nilearn nifti maskers
            are located. Used to save fitting time.

        task: str (Default = 'memory')
            Name of the experimental task
            (i.e. 'rest' is also valid).

        space: str (Default = 'MNI152NLin2009cAsym')
            Name of the template used during resampling.
            Most likely corresponding to a valid TemplateFlow name.

        output_type: str (Default = 'preproc')
            Name of the desired FMRIPrep output type. Most likely
            corresponding to a valid FMRIPrep output type name.

        extension: str (Default = '.nii.gz')
            Nifti files extension. The leading '.' is required.

        modality: str (Default = 'bold')
            Scanning modality used during the experiment.

        sub_id: str (Default = '*')
            Participant identifier. By default, returns all
            participants. The leading 'sub-' must be omitted.
            If the identifier is numeric, it should be quoted.

        ses_id: str (Default = '*')
            Session identifier. By default, returns all sessions.
            If the identifier is numeric, it should be quoted.
            The leading 'ses-' must be omitted.

    Returns: list
        Sorted list of the desired nifti file paths.

    Notes:
        All parameters excepting ``topdir`` and ``events_dir``
        can be replaced by '*' (Default), which is
        equivalent to a UNIX ``find`` pattern.
    """

    from cimaq_decoding_params import _params

    # Generate regex and glob patterns
    bold_pattern = '_'.join([f'sub-{sub_id}', f'ses-{ses_id}', f'task-{task}',
                             f'space-{space}', f'desc-{output_type}',
                             f'{modality}{extension}'])
    ev_pattern = f'sub-{sub_id}_ses-{ses_id}_task-{task}_events.tsv'

    # Load fMRI and events files paths into lists
    bold_paths = sorted(map(str, Path(topdir).rglob(f'*{bold_pattern}')))
    event_paths = sorted(map(str, (Path(events_dir).rglob(f'*{ev_pattern}'))))
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
                                  'events_path', 'behav_path',
                                  'sub_id', 'ses_id', 'task', 'space'],
                                 (item+Path(item[0]).parts[-4:-2] +
                                  ('task-'+task, 'space-'+space)))))
                for item in zipped_paths]
    
    # Setting default keyword arguments and parameters
    [session.update(**_params) for session in sessions]
    
#     def MaskerStr(session) -> str:
#         while True:
#             yield f'{session.sub_id}_{session.ses_id}*.pickle'
    
#     def MaskerPathGen(session):
#         while True:
#             MaskerStr_ = next(MaskerStr(session))
#             yield sorted(Path(MaskerStr_).rglob(MaskerStr_)).__iter__()

    if masker_dir is not None:
        [session.update({'masker_path':
                         sorted(Path(masker_dir).rglob(f'{session.sub_id}_{session.ses_id}*.pickle'))[0]})
         for session in sessions]
    else:
        [session.update({'masker_path': None}) for session in sessions]
    
    return sessions


def fetch_fmriprep_session(fmri_path: Union[str, PathLike, PosixPath] = None,
                           events_path: Union[str, PathLike, PosixPath] = None,
                           masker_path: Union[str, PathLike, PosixPath] = None,
                           strategy: str = 'Minimal',
                           task: str = 'memory',
                           space: str = 'MNI152NLin2009cAsym',
                           anat_mod: str = 'T1w',
                           session: Union[dict, Bunch] = None,
                           n_sessions: int = 1,
                           dimension: int = 1024,
                           resolution_mm: int = 3,
                           data_dir: Union[str, PathLike, PosixPath] = None,
                           sub_id: str = None,
                           ses_id: str = None,
                           lc_kws: dict = None,
                           apply_kws: dict = None,
                           clean_kws: dict = None,
                           design_kws: dict = None,
                           glm_kws: dict = None,
                           masker_kws: dict = None,
                           **kwargs
                           ) -> Bunch:
    """
    Fetch and load in memory a participant's fMRI data for a session.

    """

    import load_confounds
    from inspect import getmembers
    from nilearn import image as nimage
    from operator import itemgetter
    from sklearn.utils import Bunch
    from pathlib import Path

    
    from cimaq_decoding_utils import get_t_r, get_frame_times
    from get_difumo import get_difumo

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
#     if kwargs is not None:
#         data_dir, dimension, resolution_mm = tuple(kwargs.values())
#     else:
#         kwargs = {}

    if (hasattr(session, 'masker_path') and
        session.masker_path is not None):
        setattr(session, 'masker', unpickle(session.masker_path))
    else:
        setattr(session, 'masker_path', None)
    
    if session.masker_path is None and data_dir is None:
        attempt = os.path.join(os.getcwd(), 'difumo_atlases',
                               str(dimension), f'{resolution_mm}mm')
        if os.path.isdir(attempt):
            data_dir=attempt
        else:
            data_dir=None

        difumo = get_difumo(dimension=dimension,
                            resolution_mm=resolution_mm,
                            data_dir=data_dir)
        masker = NiftiMapsMasker(maps_img=difumo.maps,
                                 mask_img=mask_img,
                                 t_r=t_r,
                                 resampling_target='mask',
                                 **session.masker_defs).fit()            

        [setattr(session, itm[0], itm[1])
         for itm in tuple(zip(['feature_labels', 'masker'],
                              [difumo.labels.difumo_names.tolist(),
                               masker]))]

    if apply_kws is not None:
        session.apply_defs.update(apply_kws)
    if clean_kws is not None:
        session.clean_defs.update(clean_kws)

    fmri_img = nimage.clean_img(fmri_img, confounds=conf,
                                t_r=t_r, mask_img=mask_img,
                                **session.clean_defs)

    # Argument definitions for each preprocessing step
    target_shape, target_affine = mask_img.shape, fmri_img.affine

    session.glm_defs.update(Bunch(mask_img=mask_img,
                                  t_r=t_r,
                                  target_shape=target_shape,
                                  target_affine=target_affine,
                                  subject_label='_'.join([session.sub_id,
                                                          session.ses_id])))

    if design_kws is not None:
        session.design_defs.update(design_kws)
    if masker_kws is not None:
        session.masker_defs.update(masker_kws)
    if glm_kws is not None:
        session.glm_defs.update(glm_kws)

    loaded_attributes = Bunch(events=events, behav=behav,
                              frame_times=frame_times,
                              t_r=t_r,
                              confounds_loader=loader, confounds=conf,
                              confounds_strategy=strategy,
                              smoothing_fwhm=session.apply_defs.smoothing_fwhm,
                              anat_img=anat_img, fmri_img=fmri_img,
                              mask_img=mask_img)
#                               masker=masker)

    default_params = Bunch(apply_defs=session.apply_defs,
                           clean_defs=session.clean_defs,
                           design_defs=session.design_defs,
                           glm_defs=session.glm_defs,
                           masker_defs=session.masker_defs)
    session.update(loaded_attributes)
    session.update(default_params)
    session.get_t_r, session.get_frame_times = get_t_r, get_frame_times
    return session


########################################################################
# Main Pipeline: Signal Cleaning and Extraction
########################################################################


def get_glm_events(events: Union[str, PathLike, PosixPath,
                                 pd.DataFrame],
                   trial_type_cols: list = None,
                   **kwargs
                   ) -> list:
    """
    Return nilearn compatible events for each syncrhonous contrasts.

    Changing 'trial_type' to 'index' to get unique trials,
    a different design matrix can be obtained for each
    computation of interest.

    Allows OneVsOneClassification.
    """
    from operator import itemgetter
    ntrials, npad = events.shape[0], len(str(events.shape[0]))
    trials_ = events.reset_index(drop=False).index.astype(
                  str).str.zfill(npad).tolist()

    if trial_type_cols is None:
        trial_type_cols = ['trial_type']

    trial_labels = [trials_] + [events[col].tolist()
                                for col in trial_type_cols]
    return [pd.DataFrame(zip(events.onset, events.duration, labels),
                         columns=['onset', 'duration', 'trial_type'])
            for labels in trial_labels]


def weightings(signals: pd.DataFrame,
               weights: pd.DataFrame,
               ) -> pd.DataFrame:
    #                weight_labels: Union[list, pd.Index] = None
    """
    Return condition-wise weighted signals DataFrame.

    Args:
        signals: pd.DataFrame

        weights: pd.DataFrame
    """

    newsignals = signals.copy(deep=True)  # .set_axis(condition_labels, axis=0)
#     if weight_labels is not None:
#         weights = weights.set_axis(weight_labels, axis=0)
    for cond in weights.index.unique():
        newsignals.loc[cond] = newsignals.loc[cond] + weights.loc[cond]
    return newsignals


########################################################################
# Main Pipeline: Trial-Based Contrast Computation
########################################################################


def get_contrasts(fmri_img: Union[str, PathLike,
                                  PosixPath, Nifti1Image],
                  events: Union[str, PathLike,
                                PosixPath, pd.DataFrame],
                  output_type: str = 'effect_size',
                  design_kws: Union[dict, Bunch] = None,
                  glm_kws: Union[dict, Bunch] = None,
                  masker_kws: Union[dict, Bunch] = None,
                  standardize: bool = True,
                  scale: bool = False,
                  scale_between: tuple = (0, 1),
                  maximize: bool = False,
                  masker: [MultiNiftiMasker, NiftiLabelsMasker,
                           NiftiMapsMasker, NiftiMasker] = None,
                  feature_labels: Union[Sequence, pd.Index] = None,
                  session=None,
                  **kwargs
                  ) -> Bunch:
    """
    Return dict-like structure containing experimental contrasts.


    Using ``nilearn.glm.first_level.FirstLevel`` object,
    contrasts are first computed trial-wise. Then, the same is done
    for each experimental condition in ``trial_type_cols`` if a
    list of string is provided.

    Args:
        fmri_img: str, PathLike, PosixPath or Nifti1Image
            In-memory or path pointing to a ``nibabel.nifti1.Nifti1Image``.

        events: : str, PathLike, PosixPath or DataFrame
            In-memory or path pointing to a ``pandas.DataFrame``.

        output_type: str (Default = 'effect_size')
            String passed to
            ``nilearn.glm.first_level.FirstLevel.compute_contrast``
            ``output_type`` parameter.

        design_kws: dict or Bunch (Deault = None)
            Dict-like mapping of keyword arguments passed to
            ``nilearn.glm.first_level.make_first_level_design_matrix``.
            If a ``session`` object is passed in the parameters,
            the value under the corresponding key is used.

        glm_kws: dict or Bunch (Deault = None)
            Dict-like mapping of keyword arguments passed to
            ``nilearn.glm.first_level.FirstLevel.__init__``.
            If a ``session`` object is passed in the parameters,
            the value under the corresponding key is used.

        masker_kws: dict or Bunch (Deault = None)
            Dict-like mapping of keyword arguments passed to
            ``masker.__init__``.
            If a ``session`` object is passed in the parameters,
            the value under the corresponding key is used.

        standardize: bool (Default = True)
            If true (by default), the extracted brain signals are
            standardized using a ``sklearn.preprocessing.StandardScaler``
            object (demeaning ans scaling to variance). It is generally
            advised to standardize data for machine-learning operations.
            See notes for documentation, tutorials and more.

        scale: bool (Default = False)
            If true, the extracted brain signals are
            scaled (between 0 and 1 by default) using a
            ``sklearn.preprocessing.MinMaxScaler`` object. It is generally
            advised to standardize data for machine-learning operations.
            See notes for documentation, tutorials and more.

        scale_between: tuple (Default = (0, 1)
            Values between which the signal should be scaled.
            Default is (0, 1) - left = min, right = max.
            Only used if ``scale`` parameter is True.

        maximize: bool (Default = False)
            If true, scale each feature by its maximum absolute value.
            From the docs of ``sklearn.preprocessing.MaxAbsScaler``:
                '[...] Scales and translates each feature individually
                such that the maximal absolute value of each feature in
                training set is 1.0. Does not shift/center the data,
                and thus does not destroy any sparsity.'

        masker: MultiNiftiMasker, NiftiLabelsMasker,
                NiftiMapsMasker or NiftiMasker (Default = None)
            Masker object from the ``nilearn.input_data`` module meant
            to perform brain signal extraction (conversion from 4D or 3D
            image to 2D data).
            If omitted, a NiftiMasker with default parameters is used.

        feature_labels: List or pd.Index (Default = None)
            List of feature names used as columns for the brain signal matrix.
            Number of labels and number of features must match.
            An error is raised otherwise.

        session: dict or Bunch (Default = None)
            Dict-like structure containing all required and/or optional
            parameters. The functions ``fetch_fmriprep_session`` and
            ``get_fmri_session`` from ``cimaq_decoding_utils``
            return a ``session`` object. It is similar to the return
            values of ``nilearn.datasets.fetch{dataset_name}`` functions.

    Returns: ``sklearn.utils.Bunch``
        Dict-like structure with the following keys:
        ['model', 'contrast_img', 'signals',
         'feature_labels', 'condition_labels']

    Notes:
        https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    """

    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.glm.first_level import FirstLevelModel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
    from sklearn.preprocessing import StandardScaler
    from cimaq_decoding_utils import get_frame_times, get_t_r

    # Parameter initialization
    design_defs, glm_defs = {}, {}
    fmri_img = nimage.image.load_img(fmri_img)
    events = [events if isinstance(events, pd.DataFrame)
              else pd.read_csv(events,
                               sep=['\t' if splitext(events)[1][1] == 't'
                                    else ','][0])][0]
    if session is not None:
        design_defs.update(session.design_defs)
        glm_defs.update(session.glm_defs)

    t_r, frame_times = get_t_r(fmri_img), get_frame_times(fmri_img)

    if design_kws is not None:
        design_defs.update(design_kws)
    if glm_kws is not None:
        glm_defs.update(glm_kws)

    # GLM initialization and contrast computation
    design = make_first_level_design_matrix(frame_times, events=events,
                                            **design_defs)

    model = FirstLevelModel(**glm_defs).fit(run_imgs=fmri_img,
                                            design_matrices=design)
    contrasts = nimage.concat_imgs([model.compute_contrast(
                    trial, output_type=output_type) for trial in
                    tqdm_(design.columns[:-1].astype(str),
                          ncols=100,
                          desc='Computing Contrasts')])

    # Brain signals extraction
    pipe_components = ((standardize, 'standardize', StandardScaler()),
                       (maximize, 'maximize', MaxAbsScaler()),
                       (scale, 'scale', MinMaxScaler(scale_between)))

    pipe_components = [item[1:] for item in
                       list(filter(lambda x: x[0], pipe_components))]
    signals = masker.transform_single_imgs(contrasts)
    if pipe_components != []:
        pipeline = Pipeline(pipe_components)
        signals = pipeline.fit_transform(signals)
    signals = pd.DataFrame(signals,
                           index=design.iloc[:, :-1].columns)

    if feature_labels is not None:
        signals.set_axis(feature_labels, axis=1, inplace=True)

    return Bunch(model=model, contrast_img=contrasts,
                 signals=signals, feature_labels=feature_labels)


def get_all_contrasts(fmri_img: Union[str, PathLike,
                                      PosixPath, Nifti1Image]=None,
                      events: Union[str, PathLike,
                                    PosixPath, pd.DataFrame]=None,
                      output_type: str = 'effect_size',
                      masker: [MultiNiftiMasker, NiftiLabelsMasker,
                               NiftiMapsMasker, NiftiMasker] = None,
                      standardize: bool = True,
                      maximize: bool = False,
                      scale: bool = False,
                      scale_between: tuple = (0, 1),
                      trial_type_cols: list = None,
                      feature_labels: Union[list, pd.Index] = None,
                      design_kws: Union[dict, Bunch] = None,
                      glm_kws: Union[dict, Bunch] = None,
                      session: Union[dict, Bunch] = None,
                      extract_only: bool = False
                      ) -> Bunch:
    """
    Return dict-like structure containing experimental contrasts.


    Using ``nilearn.glm.first_level.FirstLevel`` object,
    contrasts are first computed trial-wise. Then, the same is done
    for each experimental condition in ``trial_type_cols`` if a
    list of string is provided. Each trial signal is weighted
    according to which experimental contrast it corresponds to.

    Args:
        fmri_img: str, PathLike, PosixPath or Nifti1Image
            In-memory or path pointing to a ``nibabel.nifti1.Nifti1Image``.

        events: : str, PathLike, PosixPath or DataFrame
            In-memory or path pointing to a ``pandas.DataFrame``.

        output_type: str (Default = 'effect_size')
            String passed to the ``output_type`` parameter from
            ``nilearn.glm.first_level.FirstLevel.compute_contrast``.

        masker: MultiNiftiMasker, NiftiLabelsMasker,
                NiftiMapsMasker or NiftiMasker (Default = None)
            Masker object from the ``nilearn.input_data`` module meant
            to perform brain signal extraction (conversion from 4D or 3D
            image to 2D data).
            If omitted, a NiftiMasker with default parameters is used.

        standardize: bool (Default = True)
            If true (by default), the extracted brain signals are
            standardized using a ``sklearn.preprocessing.StandardScaler``
            object (demeaning ans scaling to variance). It is generally
            advised to standardize data for machine-learning operations.
            See notes for documentation, tutorials and more.

        scale: bool (Default = True)
            If true (by default), the extracted brain signals are
            scaled (between 0 and 1 by default) using a
            ``sklearn.preprocessing.MinMaxScaler`` object. It is generally
            advised to standardize data for machine-learning operations.
            See notes for documentation, tutorials and more.

        scale_between: tuple (Default = (0, 1)
            Values between which the signal should be scaled.
            Default is (0, 1) - left = min, right = max.
            Only used if ``scale`` parameter is True.

        maximize: bool (Default = False)
            If true, scale each feature by its maximum absolute value.
            From the docs of ``sklearn.preprocessing.MaxAbsScaler``:
                '[...] Scales and translates each feature individually
                such that the maximal absolute value of each feature in
                training set is 1.0. Does not shift/center the data,
                and thus does not destroy any sparsity.'

        trial_type_cols: list
            List of strings representing the differents experimental
            conditions which are not labeled 'trial_type' in the
            events DataFrame. This parameter is passed to the
            ``cimaq_decoding_utils.get_glm_events`` function.

        design_kws: dict or Bunch (Deault = None)
            Dict-like mapping of keyword arguments passed to
            ``nilearn.glm.first_level.make_first_level_design_matrix``.
            If a ``session`` object is passed in the parameters,
            the value under the corresponding key is used.

        glm_kws: dict or Bunch (Deault = None)
            Dict-like mapping of keyword arguments passed to
            ``nilearn.glm.first_level.FirstLevel.__init__``.
            If a ``session`` object is passed in the parameters,
            the value under the corresponding key is used.

        feature_labels: List or pd.Index (Default = None)
            List of feature names used as columns in the brain signal matrix.
            Number of labels and number of features must match.
            An error is raised otherwise.

        session: dict or Bunch (Default = None)
            Dict-like structure containing all required and/or optional
            parameters. The functions ``fetch_fmriprep_session`` and
            ``get_fmri_session`` from ``cimaq_decoding_utils``
            return a ``session`` object. It is similar to the return
            values of ``nilearn.datasets.fetch{dataset_name}`` functions.

    Returns: ``sklearn.utils.Bunch``
        Dict-like structure with the following keys:
        ['model', 'contrast_img', 'signals',
         'feature_labels', 'condition_labels']


    Notes:
        https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
    """
    
    from operator import itemgetter
    from os.path import splitext
    from cimaq_decoding_utils import get_frame_times, get_t_r
    
    if session is not None:
        fmri_img, events, masker = itemgetter(*['fmri_img',
                                                'events',
                                                'masker'])(session)
        design_kws, glm_kws = itemgetter(*['design_defs',
                                           'glm_defs'])(session)


    fmri_img = nimage.image.load_img(fmri_img)
    events = [events if isinstance(events, pd.DataFrame)
              else pd.read_csv(events,
                               sep=['\t' if splitext(events)[1][1] == 't'
                                    else ','][0])][0]

    if trial_type_cols is None:
        trial_type_cols = []
        events = events.reset_index(drop=False).rename({'index': 'whole'})

    glm_events = get_glm_events(events, trial_type_cols)
    get_contrasts_params = dict(masker=masker, standardize=standardize,
                                scale=scale, output_type=output_type,
                                design_kws=design_kws, glm_kws=glm_kws,
                                feature_labels=feature_labels)
    contrasts_dict = Bunch(**dict(((col[1],
                                    get_contrasts(fmri_img=fmri_img,
                                                  events=glm_events[col[0]],
                                                  **get_contrasts_params)))
                                  for col in
                                  enumerate(['whole'] + trial_type_cols)))

    # Make copies of basic condition contrasts (trial-by-trial)
    # for each different trial labelling in ``trial_type_cols``
    matrices = [contrasts_dict.whole.signals.copy(deep=True).set_axis(
                    events_.trial_type, axis=0)
                for events_ in glm_events]
    # Gather individually extracted signals in one place
    weights = [contrasts_dict[key].signals for key in
               list(key for key in contrasts_dict.keys()
                    if key != 'whole')]
    # Apply its respective trial-wise weights to each matrice in ``matrices``
    weighted_matrices = [weightings(item[0], item[1]) for item in
                        tuple(zip(matrices[1:], weights))]
    # Sum previous step matrices with the first basic (trial-by-trial) contrasts
    weighted = pd.DataFrame(np.sum(np.array(weighted_matrices), axis=0),
                            index=matrices[0].index,
                            columns=matrices[0].columns)
    
    if extract_only is True:
        return weighted

#     weighted_matrices = [mat[1].set_axis(glm_events[mat[0]].trial_type)
#                          for mat in enumerate([weighted]*len(glm_events))]

    contrasts_dict.update(Bunch(**{'matrices': matrices,
                                   'weights': weights,
                                   'weighted_matrices': weighted_matrices,
                                   'trial_type_cols': trial_type_cols,
                                   'signal_matrix': weighted}))

    return Bunch(**contrasts_dict)


########################################################################
# Main Pipeline: Model Evaluation and Classifier Validation
########################################################################


def validate_model(estimator,
                   X: Iterable, y: Iterable,
                   test_size: float = 0.8,
                   cv: Union[int, callable] = None,
                   stratify: Iterable = None,
                   random_state: int = None,
                   **kwargs
                   ) -> pd.DataFrame:

    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_predict

    validation_params = dict(test_size=test_size, shuffle=True,
                             stratify=stratify, random_state=None)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        **validation_params)

    estimator.fit(X_train, y_train)
    cv_score = cross_val_predict(estimator, X_test, y_test,
                                 groups=y_test, cv=cv)
    cr_test = pd.DataFrame(classification_report(y_pred=cv_score,
                                                 y_true=y_test,
                                                 output_dict=True,
                                                 zero_division=0))
    return cr_test


########################################################################
# Main Pipeline: Dimensionality Reduction
########################################################################


def untangle(X: Iterable,
             y: Iterable,
             n_clusters: int = None,
             get_connectivity: bool = True,
             compute_distances: bool = True,
             kind: str = 'correlation',
             agglo_kws: Union[dict, Bunch] = None
             ) -> FeatureAgglomeration:

    from nilearn.connectome import ConnectivityMeasure as CM
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.covariance import LedoitWolf
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_classif

    agglo_defs = dict(affinity='euclidean',
                      compute_full_tree='auto',
                      linkage='ward',
                      pooling_func=np.mean,
                      distance_threshold=None,
                      compute_distances=compute_distances)
    
    if get_connectivity is True:
        connect_mat = CM(LedoitWolf(),
                         kind='correlation').fit_transform([X.values])[0]
    else:
        connect_mat = None

    if n_clusters is None:
        n_clusters = divmod(X.shape[1], 2)[0] - 1
        if n_clusters == 0:
            n_clusters = 1

    if agglo_kws is None:
        agglo_kws = {}
    agglo_defs.update(agglo_kws)

    agglo = FeatureAgglomeration(n_clusters=n_clusters,
                                 connectivity=connect_mat,
                                 **agglo_defs)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    agglo.fit(X, y)

    setattr(agglo, 'cluster_indexes_',
            pd.DataFrame(zip(agglo.labels_,
                             agglo.feature_names_in_),
                         columns=['cluster',
                                  'feature']).groupby('cluster').feature)

    skb = SelectKBest(k=1, score_func=mutual_info_classif)
    factor_leaders = [skb.fit(X[itm[1]], y).get_feature_names_out()[0]
                      for itm in tuple(agglo.cluster_indexes_)]
    setattr(agglo, 'factor_leaders', factor_leaders)
    return agglo

