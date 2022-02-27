#!/usr/bin/python3


########################################################################
# Unused yet
########################################################################


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
