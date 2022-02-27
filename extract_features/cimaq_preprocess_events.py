#!/usr/bin/python3

import os
import pandas as pd
import pathlib

from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img
from os import PathLike
from pathlib import Path, PosixPath
from sklearn.utils import Bunch
from typing import Union

########################################################################
# CIMA-Q-Specific
########################################################################


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
        if hasattr(session, 'fmri_img'):
            events, fmri_img = session.events, session.fmri_img
        else:
            fmri_img = load_img(session.fmri_path)
            events = pd.read_csv(session.events_path, sep='\t')
    else:
        fmri_img = load_img(fmri_img)
        events = [events if isinstance(events, pd.DataFrame)
                  else pd.read_csv(events, sep='\t')][0]
    t_r, s_task = fmri_img.header.get_zooms()[-1], events.copy(deep=True)
    s_task['duration'] = s_task['duration']+s_task['isi']
    s_task.stim_file.fillna('empty_box_gris.bmp', inplace=True)
    s_task.stim_category.fillna('ctl', inplace=True)
    s_task.stim_id.fillna('ctl', inplace=True)
    scanDur, numCol = fmri_img.shape[-1]*t_r, s_task.shape[1]
    s_task['trial_ends'] = s_task.onset+s_task.duration.values
    s_task['unscanned'] = (s_task.trial_ends > scanDur).astype(int)
    s_task = s_task[s_task['unscanned'] == 0]
    # Special contrasts
    s_task['recognition_performance'] = [row[1].recognition_performance
                                         if row[1].trial_type == 'Enc'
                                         else 'Ctl' for row
                                         in s_task.iterrows()]
    s_task['ctl_miss_ws_cs'] = ['Cs' if
                                'Hit' == row[1].recognition_performance
                                else 'Ws' if
                                bool(row[1].recognition_accuracy and not
                                     row[1].position_accuracy)
                                else row[1].recognition_performance
                                for row in s_task.iterrows()]
    s_task.duration = s_task.duration+s_task.isi
    s_task.iloc[:, -1].rename({s_task.iloc[:, -1].name: 'position_performance'},
                              axis=1, inplace=True)
    return s_task
