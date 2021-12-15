#!/usr/bin/python3

import os
import pandas as pd
from typing import Union

def fetch_fmriprep_session(fmri_path:Union[str,os.PathLike],
                           events_dir:Union[str,os.PathLike],
                           strategy:str='Minimal',
                           task:str='memory',
                           space:str='MNI152NLin2009cAsym',
                           anat_mod:str='T1w',
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
    from cimaq_decoding_utils import get_fmriprep_anat, get_fmriprep_mask
    from cimaq_decoding_utils import clean_fmri, get_events, get_behav
    from cimaq_decoding_utils import get_tr, get_frame_times

    events_path = get_events(fmri_path, events_dir)
    if events_path is False:
        return False
    events = pd.read_csv(events_path, sep='\t')
    behav_path = get_behav(fmri_path, events_dir)
    behav = pd.read_csv(behav_path, sep='\t')
    sub_id, ses_id = Path(fmri_path).parts[-4:-2]
    mask_path = get_fmriprep_mask(fmri_path)
    anat_path = get_fmriprep_anat(fmri_path)
    loader = dict(getmembers(load_confounds))[f'{strategy}']
    loader = [loader(**lc_kws) if lc_kws is not None
              else loader()][0]
    conf = loader.load(fmri_path)
    fmri_img, mask_img, anat_img = tuple(map(nimage.load_img,
                                             [fmri_path, mask_path,
                                              anat_path]))
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

    return Bunch(sub_id=sub_id, ses_id=ses_id, task=task, space=space,
                 fmri_path=fmri_path, mask_path=mask_path,
                 events_path=events_path, behav_path=behav_path,
                 events=events, behav=behav, frame_times=frame_times,
                 confounds_loader=loader, confounds=conf, confounds_strategy=strategy,
                 smoothing_fwhm=_params.apply_defs.smoothing_fwhm,
                 fmri_img=fmri_img, mask_img=mask_img,
                 cleaned_fmri=cleaned_fmri,
                 **Bunch(clean_defs=_params.clean_defs,
                         design_defs=_params.design_defs,
                         glm_defs=_params.glm_defs,
                         masker_defs=_params.masker_defs))
  