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
import nistats
import nilearn
import scipy
import nibabel

from numpy import nan as NaN
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nibabel.nifti1 import Nifti1Image
from sklearn.utils import Bunch
from typing import Union

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Create trial-unique brain maps of beta coefficients
        Input: Folders with task file, confounds and fMRI data
        """)

    parser.add_argument(
        "-s", "--sdir",
        required=True, nargs="+",
        help="Path to id_list.tsv, a list of of subject ids",
        )

    parser.add_argument(
        "-t", "--tdir",
        required=True, nargs="+",
        help="Folder with task files",
        )

    parser.add_argument(
        "-m", "--mdir",
        required=True, nargs="+",
        help="Folder with motion files",
        )

    parser.add_argument(
        "-f", "--fdir",
        required=True, nargs="+",
        help="Folder with fMRI files",
        )

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output    folder - if doesnt exist it will be created",
        )

    parser.add_argument(
        "-v", "--verbose",
        required=False, nargs="+",
        help="Verbose to get more information about what's going on",
        )

    args =  parser.parse_args()
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def get_task_Files(slist, tDir):
    """ Returns a list of task files (one file per participant)
    for participants whose dccid is listed in slist
    Parameter:
    ----------
    slist: a list of strings (dccids of subjects to include)
    task_files: list of strings (paths to task files)

    Return:
    ----------
    None (beta maps are saved directly in outdir)
    """
    # TEST IF THIS PART WORKS: only run subjects if id is on list (argument)
    if not os.path.exists(tDir):
        sys.exit('This folder does not exist: {}'.format(tDir))
        return
    all_files = glob.glob(os.path.join(tDir,'sub-*events.tsv'))
    subs = slist.index
    task_files = []
    for tfile in all_files:
        id = os.path.basename(tfile).split('-')[1].split('_')[0]
        if int(id) in subs:
            print(id)
            task_files.append(tfile)

    return task_files

def get_confounds(id, mDir, short_conf):
    """ Imports a single subject's *confounds.tsv file
    and returns a pandas dataframe of regressors (motion,
    slow drift, white matter intensity, frames to scrub, etc.)
    Parameters:
    ----------
    id: string (participant's dccid, Loris identifier)
    mDir: string (path to directory with *confounds.tsv files)
    short_conf: boolean (determines which confounds to use in model:
        False : full set of confounds outputed by NIAK (e.g., slow drift...);
        True : partial set of Niak confounds, rest is modelled in Nistats )

    Return:
    ----------
    confounds: pandas dataframe (regressors for first-level_model)
    """
    mfile = glob.glob(os.path.join(mDir, 'fmri_sub'+id+'*confounds.tsv'))
    if len(mfile) == 0:
        print('*confounds.tsv file missing for participant '+id)
        return
    else:
        confounds = pd.read_csv(mfile[0], sep='\t')

        if short_conf == True:
            colsKeep = ['motion_tx', 'motion_ty', 'motion_tz', 'motion_rx',
            'motion_ry', 'motion_rz', 'scrub', 'wm_avg', 'vent_avg']

            confounds = confounds[colsKeep]
    return confounds

def extract_events(taskFile, id, outdir, scanDur):
    """Loads a single subject's task file and returns a pandas
    dataframe that specifies trial onsets, duration and conditions.
    Also exports vectors of trial labels as .tsv files (used for
    classification analyses).
    Parameters:
    ----------
    subFile: string (path to subject's task file)
    id: string (subject's dccid)
    outdir: string (ouput directory)
    scanDur: scan's duration in seconds

    Return:
    ----------
    tData: pandas dataframe (events parameters for first-level model)
    """
    tData = pd.read_csv(taskFile, sep='\t')

    # rename "trial_type" column as "condition"
    tData.rename(columns={'trial_type':'condition'}, inplace=True)

    # Add columns to dataframe
    numCol = tData.shape[1] # number of columns
    insertCol = [4, numCol+1, numCol+2, numCol+3]
    colNames = ['trial_type', 'unscanned', 'ctl_miss_hit', 'ctl_miss_ws_cs']
    colVals = ['TBD', 0, 'TBD', 'TBD']
    for i in range(0, len(colNames)):
        tData.insert(loc=insertCol[i], column=colNames[i], value=colVals[i], allow_duplicates=True)

    # The 'unscanned' column flag trials for which
    # no brain data was acquired : The scan's duration
    # is shorter than the trial's offset time (0 = data, 1 = no data)
    for j in tData[tData['offset']>scanDur].index:
        tData.loc[j, 'unscanned']=1

    # pad trial numbers with zeros (on the left) to preserve trial
    # temporal order when trials are alphabetized
    tData['trial_number'] = tData['trial_number'].astype('object', copy=False)
    for k in tData.index:
        tData.loc[k, 'trial_number'] = str(tData.loc[k, 'trial_number']).zfill(3)

    # Fill trial_type column, and columns that identify missed, wrong source
    # and correct source trials.
    # The "trial_type" column must contain a different entry per trial
    # to model trials separately (with own beta map) in Nistats.
    countEnc = 0 # number of encoding trials (normally 78)
    countCTL = 0 # number of control trials (normally 39)
    for m in tData[tData['condition']=='Enc'].index:
        countEnc = countEnc + 1
        tData.loc[m, 'trial_type'] = 'Enc'+str(countEnc)
        if tData.loc[m, 'position_accuracy'] == 0:
            tData.loc[m, 'ctl_miss_hit']='missed'
            tData.loc[m, 'ctl_miss_ws_cs']='missed'
        elif tData.loc[m, 'position_accuracy'] == 1:
            tData.loc[m, 'ctl_miss_hit']='hit'
            tData.loc[m, 'ctl_miss_ws_cs']='wrongsource'
        elif tData.loc[m, 'position_accuracy'] == 2:
            tData.loc[m, 'ctl_miss_hit']='hit'
            tData.loc[m, 'ctl_miss_ws_cs']='correctsource'
    for n in tData[tData['condition']=='CTL'].index:
        countCTL = countCTL + 1
        tData.loc[n, 'trial_type'] = 'CTL'+str(countCTL)
        tData.loc[n, 'ctl_miss_hit']='control'
        tData.loc[n, 'ctl_miss_ws_cs']='control'

    #save extended task file dataframe as .tsv file
    tData.to_csv(outdir+'/sub-'+id+'_events.tsv',
    sep='\t', header=True, index=False)

    #keep only trials for which fMRI data was collected
    tData = tData[tData['unscanned']==0]

    #Save vectors of trial labels (e.g., encoding vs control)
    #to label trials for classification in later analyses
    vec1 = tData['condition']
    vec1.to_csv(outdir+'/sub-'+id+'_enco_ctl.tsv',
    sep='\t', header=True, index=False)

    vec2 = tData['ctl_miss_hit']
    vec2.to_csv(outdir+'/sub-'+id+'_ctl_miss_hit.tsv',
    sep='\t', header=True, index=False)

    vec3 = tData['ctl_miss_ws_cs']
    vec3.to_csv(outdir+'/sub-'+id+'_ctl_miss_ws_cs.tsv',
    sep='\t', header=True, index=False)

    # Only keep columns needed to build a design matrix
    # to input a first-level model in nistats
    event_cols = ['onset', 'duration', 'trial_type', 'condition', 'ctl_miss_hit',
    'ctl_miss_ws_cs', 'trial_number']

    tData = tData[event_cols]

    return tData


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


def extract_betas(taskList, taskdir, outdir, motiondir, fmridir):
    """
    Extracts beta maps for each subject whose task file is in taskList.

    Parameters:
    ----------
    taskList: list of strings (each a path to a task file)
    taskdir: string (path to directory with task files, where save labels)
    outdir: string (path to output directory)
    motiondir: sring (path to directory with *confounds.tsv files)
    fmridir: string (path to directorty for 4D fMRI .nii files)

    Return:
    ----------
    None (beta maps are saved directly in outdir)
    """
    b_outdir = os.path.join(outdir, 'features', 'beta_maps')
    if not os.path.exists(b_outdir):
        os.mkdir(b_outdir)

    ev_outdir = os.path.join(outdir, 'task_files', 'events')
    if not os.path.exists(ev_outdir):
        os.mkdir(ev_outdir)

    for tfile in taskList:
        sub_id = os.path.basename(tfile).split('-')[1].split('_')[0]
        confounds = get_confounds(sub_id, motiondir, False)
        scanDur = confounds.shape[0]*2.5 #CIMAQ fMRI TR = 2.5s
        events = extract_events(tfile, sub_id, ev_outdir, scanDur)
        get_subject_betas(sub_id, confounds, events, fmridir, b_outdir)
    return

def main():
    args =  get_arguments()
    # sub_list.tsv, a list of subject dccids in .tsv format
    slist = pd.read_csv(args.sdir[0], sep = '\t')
    slist.set_index('sub_ids', inplace=True)

    task_dir = args.tdir[0]
    task_files = get_task_Files(slist, task_dir)
    motion_dir = args.mdir[0]
    fmri_dir = args.fdir[0]
    output_dir = args.odir[0]
    extract_betas(task_files, task_dir, output_dir, motion_dir, fmri_dir)

if __name__ == '__main__':
    sys.exit(main())
