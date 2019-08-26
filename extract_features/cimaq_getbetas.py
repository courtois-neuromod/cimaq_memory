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
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel

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
    tDir: a string (path to directory with task files)

    Return:
    ----------
    task_files (a list of strings: paths to task files)
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

def sub_tcontrasts1(id, tr, frame_times, hrf_model, confounds,
all_events, fmri_img, sub_outdir):
    """Uses nistats first-level model to create maps of beta values
    that correspond to the following contrasts between conditions:
    control, encoding, and encoding_minus_control
    Parameters:
    ----------
    id: string (subject's dccid)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmridir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """
    # Model 1: encoding vs control conditions
    events1 = all_events.copy(deep = True)
    cols = ['onset', 'duration', 'condition']
    events1 = events1[cols]
    events1.rename(columns={'condition':'trial_type'}, inplace=True)

    # create the model
    model1 = FirstLevelModel(t_r=tr, drift_model = None, standardize = True,
    noise_model='ar1', hrf_model = hrf_model)
    # Should data be standardized?

    # create the design matrices
    design1 = make_first_level_design_matrix(frame_times, events=events1,
                                            drift_model=None, add_regs=confounds,
                                            hrf_model=hrf_model)

    # fit model with design matrix
    model1 = model1.fit(fmri_img, design_matrices = design1)
    design_matrix1 = model1.design_matrices_[0]

    # Condition order: control, encoding (alphabetical)
    # contrast 1.1: control condition
    ctl_vec = np.repeat(0, design_matrix1.shape[1])
    ctl_vec[0] = 1
    b11_map = model1.compute_contrast(ctl_vec, output_type='effect_size') #"effect_size" for betas
    b11_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_ctl.nii')
    nibabel.save(b11_map, b11_name)

    #contrast 1.2: encoding condition
    enc_vec = np.repeat(0, design_matrix1.shape[1])
    enc_vec[1] = 1
    b12_map = model1.compute_contrast(enc_vec, output_type='effect_size') #"effect_size" for betas
    b12_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_enc.nii')
    nibabel.save(b12_map, b12_name)

    #contrast 1.3: encoding minus control
    encMinCtl_vec = np.repeat(0, design_matrix1.shape[1])
    encMinCtl_vec[1] = 1
    encMinCtl_vec[0] = -1
    b13_map = model1.compute_contrast(encMinCtl_vec, output_type='effect_size') #"effect_size" for betas
    b13_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_enc_minus_ctl.nii')
    nibabel.save(b13_map, b13_name)

    return

def sub_tcontrasts2(id, tr, frame_times, hrf_model, confounds,
all_events, fmri_img, sub_outdir):
    """Uses nistats first-level model to create maps of beta values
    that correspond to the following contrasts between conditions:
    hit, miss, hit_minus_miss, hit_minus_ctl and miss_minus_ctl
    Parameters:
    ----------
    id: string (subject's dccid)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmridir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """
    # Model 1: encoding vs control conditions
    events2 = all_events.copy(deep = True)
    cols = ['onset', 'duration', 'ctl_miss_hit']
    events2 = events2[cols]
    events2.rename(columns={'ctl_miss_hit':'trial_type'}, inplace=True)

    # create the model
    model2 = FirstLevelModel(t_r=tr, drift_model = None, standardize = True,
    noise_model='ar1', hrf_model = hrf_model)
    # Should data be standardized?

    # create the design matrices
    design2 = make_first_level_design_matrix(frame_times, events=events2,
    drift_model=None, add_regs=confounds, hrf_model=hrf_model)

    # fit model with design matrix
    model2 = model2.fit(fmri_img, design_matrices = design2)
    design_matrix2 = model2.design_matrices_[0]

    # Condition order: control, hit, missed (alphabetical)
    #contrast 2.1: miss
    miss_vec = np.repeat(0, design_matrix2.shape[1])
    miss_vec[2] = 1
    b21_map = model2.compute_contrast(miss_vec, output_type='effect_size') #"effect_size" for betas
    b21_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_miss.nii')
    nibabel.save(b21_map, b21_name)

    #contrast 2.2: hit
    hit_vec = np.repeat(0, design_matrix2.shape[1])
    hit_vec[1] = 1
    b22_map = model2.compute_contrast(hit_vec, output_type='effect_size') #"effect_size" for betas
    b22_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_hit.nii')
    nibabel.save(b22_map, b22_name)

    #contrast 2.3: hit minus miss
    hit_min_miss_vec = np.repeat(0, design_matrix2.shape[1])
    hit_min_miss_vec[1] = 1
    hit_min_miss_vec[2] = -1
    b23_map = model2.compute_contrast(hit_min_miss_vec, output_type='effect_size') #"effect_size" for betas
    b23_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_hit_minus_miss.nii')
    nibabel.save(b23_map, b23_name)

    #contrast 2.4: hit minus control
    hit_min_ctl_vec = np.repeat(0, design_matrix2.shape[1])
    hit_min_ctl_vec[1] = 1
    hit_min_ctl_vec[0] = -1
    b24_map = model2.compute_contrast(hit_min_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b24_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_hit_minus_ctl.nii')
    nibabel.save(b24_map, b24_name)

    #contrast 2.5: miss minus control
    miss_min_ctl_vec = np.repeat(0, design_matrix2.shape[1])
    miss_min_ctl_vec[2] = 1
    miss_min_ctl_vec[0] = -1
    b25_map = model2.compute_contrast(miss_min_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b25_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_miss_minus_ctl.nii')
    nibabel.save(b25_map, b25_name)

    return

def sub_tcontrasts3(id, tr, frame_times, hrf_model, confounds,
all_events, fmri_img, sub_outdir):
    """Uses nistats first-level model to create maps of beta values
    that correspond to the following contrasts between conditions:
    correctsource (cs), wrongsource (ws), cs_minus_ws, cs_minus_miss,
    ws_minus_miss, cs_minus_ctl, ws_minus_ctl
    hit, miss, hit_minus_miss, hit_minus_ctl and miss_minus_ctl
    Parameters:
    ----------
    id: string (subject's dccid)
    tr: float (length of time to repetition, in seconds)
    frames_times: list of float (onsets of fMRI frames, in seconds)
    hrf_model: string (type of HRF model)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmridir: string (path to directory with fMRI data)
    outdir: string (path to subject's image output directory)

    Return:
    ----------
    None (beta maps are exported in sub_outdir)
    """
    # Model 1: encoding vs control conditions
    events3 = all_events.copy(deep = True)
    cols = ['onset', 'duration', 'ctl_miss_ws_cs']
    events3 = events3[cols]
    events3.rename(columns={'ctl_miss_ws_cs':'trial_type'}, inplace=True)

    # create the model
    model3 = FirstLevelModel(t_r=tr, drift_model = None, standardize = True,
    noise_model='ar1', hrf_model = hrf_model)
    # Should data be standardized?

    # create the design matrices
    design3 = make_first_level_design_matrix(frame_times, events=events3,
                                            drift_model=None, add_regs=confounds,
                                            hrf_model=hrf_model)

    # fit model with design matrix
    model3 = model3.fit(fmri_img, design_matrices = design3)
    design_matrix3 = model3.design_matrices_[0]

    # Condition order: control, correct source, missed, wrong source (alphabetical)
    #contrast 3.1: wrong source
    ws_vec = np.repeat(0, design_matrix3.shape[1])
    ws_vec[3] = 1
    b31_map = model3.compute_contrast(ws_vec, output_type='effect_size') #"effect_size" for betas
    b31_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_ws.nii')
    nibabel.save(b31_map, b31_name)

    #contrast 3.2: correct source
    cs_vec = np.repeat(0, design_matrix3.shape[1])
    cs_vec[1] = 1
    b32_map = model3.compute_contrast(cs_vec, output_type='effect_size') #"effect_size" for betas
    b32_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_cs.nii')
    nibabel.save(b32_map, b32_name)

    #contrast 3.3: correct source minus wrong source
    cs_minus_ws_vec = np.repeat(0, design_matrix3.shape[1])
    cs_minus_ws_vec[1] = 1
    cs_minus_ws_vec[3] = -1
    b33_map = model3.compute_contrast(cs_minus_ws_vec, output_type='effect_size') #"effect_size" for betas
    b33_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_cs_minus_ws.nii')
    nibabel.save(b33_map, b33_name)

    #contrast 3.4: correct source minus miss
    cs_minus_miss_vec = np.repeat(0, design_matrix3.shape[1])
    cs_minus_miss_vec[1] = 1
    cs_minus_miss_vec[2] = -1
    b34_map = model3.compute_contrast(cs_minus_miss_vec, output_type='effect_size') #"effect_size" for betas
    b34_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_cs_minus_miss.nii')
    nibabel.save(b34_map, b34_name)

    #contrast 3.5: wrong source minus miss
    ws_minus_miss_vec = np.repeat(0, design_matrix3.shape[1])
    ws_minus_miss_vec[3] = 1
    ws_minus_miss_vec[2] = -1
    b35_map = model3.compute_contrast(ws_minus_miss_vec, output_type='effect_size') #"effect_size" for betas
    b35_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_ws_minus_miss.nii')
    nibabel.save(b35_map, b35_name)

    #contrast 3.6: correct source minus control
    cs_minus_ctl_vec = np.repeat(0, design_matrix3.shape[1])
    cs_minus_ctl_vec[1] = 1
    cs_minus_ctl_vec[0] = -1
    b36_map = model3.compute_contrast(cs_minus_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b36_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_cs_minus_ctl.nii')
    nibabel.save(b36_map, b36_name)

    #contrast 3.7: wrong source minus control
    ws_minus_ctl_vec = np.repeat(0, design_matrix3.shape[1])
    ws_minus_ctl_vec[3] = 1
    ws_minus_ctl_vec[0] = -1
    b37_map = model3.compute_contrast(ws_minus_ctl_vec, output_type='effect_size') #"effect_size" for betas
    b37_name = os.path.join(sub_outdir, 'betas_sub'+str(id)+'_ws_minus_ctl.nii')
    nibabel.save(b37_map, b37_name)

    return

def get_subject_betas(id, confounds, all_events, fmridir, outdir):
    """Uses nistats first-level model to create and export maps of beta values
    for a single subject, using events and confounds dataframes as parameters.

    The nistats.first_level_model is an interface to use the glm
    implemented in nistats.regression

    Parameters:
    ----------
    id: string (subject's dccid)
    confounds: pandas dataframe (motion and other noise regressors)
    all_events: string (task information: trials' onset time, duration and label)
    fmridir: string (path to directory with fMRI data)
    outdir: string (path to output directory)

    Return:
    ----------
    None (beta maps are exported in outdir)
    """
    #Create output directory for subject's images
    sub_outdir = os.path.join(outdir, str(id))
    if not os.path.exists(sub_outdir):
        os.mkdir(sub_outdir)

    trial_outdir = os.path.join(sub_outdir, 'TrialContrasts')
    if not os.path.exists(trial_outdir):
        os.mkdir(trial_outdir)

    scanfile = glob.glob(os.path.join(fmridir, 'fmri_sub'+id+'*nii'))
    if len(scanfile) == 0:
        print('fMRI data file missing for participant '+id)
        return
    elif len(scanfile) < 0:
        print('multiple fMRI data files for participant '+id)
        return
    else:
        # Apply 8mm smoothing since using unsmoothed preprocessed data
        # (from NIAK's 'resample' output directory)
        fmri_imgNS = scanfile[0]
        fmri_img = image.smooth_img(fmri_imgNS, 8)

        tr = 2.5  # CIMAQ fMRI = 2.5s TRs
        n_scans = confounds.shape[0] #number of frames
        frame_times = np.arange(n_scans)*tr # corresponding frame times
        hrf_model = 'spm' # alternatives: 'glover' or 'spm + derivative'

        # 117 trials if full scan (no missing frames)
        numTrials = all_events.shape[0]

        # Home-made concatenation, as backup
        # Compile temporally ordered list of betas (per trial)
        all_betas_filelist = []

        # Create a design matrix, first level model and beta map for
        # each encoding and control trial
        # Slow drift modelled with NIAK preprocessing pipeline rather than Nistats
        for i in range (0, numTrials):

            # copy all_events dataframe to keep the original intact
            events = all_events.copy(deep = True)

            # Determine trial number and condition (encoding or control)
            tnum = events.iloc[i, 6]
            currentCondi = events.iloc[i, 3]
            tname = events.iloc[i, 2] #e.g., Enc3, CTL31

            # modify trial_type column to model only the trial of interest
            for j in events.index:
                if events.loc[j, 'trial_number'] != tnum:
                    events.loc[j, 'trial_type']= 'X_'+ events.loc[j, 'condition']
                    # X for condition to remain in alphabetical order: trial of interest, X_CTL, X_Enc
            # design matrix columns that correspond to task conditions are
            # ordered alphabetically (by name of condition)

            # remove unecessary columns
            cols = ['onset', 'duration', 'trial_type']
            events = events[cols]

            # create the model
            trial_model = FirstLevelModel(t_r=tr, drift_model = None, standardize = True, noise_model='ar1',
                                       hrf_model = hrf_model)
            # Should data be standardized?

            # create the design matrix
            design = make_first_level_design_matrix(frame_times, events=events,
                                                    drift_model=None, add_regs=confounds,
                                                    hrf_model=hrf_model)

            # fit model with design matrix
            trial_model = trial_model.fit(fmri_img, design_matrices = design)

            design_matrix = trial_model.design_matrices_[0]

            # Contrast vector: 1 in design matrix column that corresponds to trial of interest, 0s elsewhere
            contrast_vec = np.repeat(0, design_matrix.shape[1])
            contrast_vec[0] = 1

            # compute the contrast's beta maps with the model.compute_contrast() method,
            # based on contrast provided.
            # https://nistats.github.io/modules/generated/nistats.first_level_model.FirstLevelModel.html
            b_map = trial_model.compute_contrast(contrast_vec, output_type='effect_size') #"effect_size" for betas
            b_name = os.path.join(trial_outdir, 'betas_sub'+str(id)+'_Trial'+str(tnum)+'_'+tname+'.nii')
            #export b_map .nii image in output directory
            nibabel.save(b_map, b_name)
            all_betas_filelist.append(b_name)

        alltrials_betas = nibabel.funcs.concat_images(images=all_betas_filelist, check_affines=True, axis=None)
        print(alltrials_betas.shape)
        nibabel.save(alltrials_betas, os.path.join(sub_outdir, 'concat_all_betas_sub'+str(id)+'.nii'))

        #Create subdirectory to save task contasts
        tc_outdir = os.path.join(sub_outdir, 'TaskContrasts')
        if not os.path.exists(tc_outdir):
            os.mkdir(tc_outdir)

        #Create beta maps for task contrasts
        #Encoding & control task
        sub_tcontrasts1(id, tr, frame_times, hrf_model, confounds,
        all_events, fmri_img, tc_outdir)
        #Control, hit and miss
        sub_tcontrasts2(id, tr, frame_times, hrf_model, confounds,
        all_events, fmri_img, tc_outdir)
        #Control, miss, wrong source and correct source
        sub_tcontrasts3(id, tr, frame_times, hrf_model, confounds,
        all_events, fmri_img, tc_outdir)

    return

def extract_betas(taskList, taskdir, outdir, motiondir, fmridir):
    """ Extracts beta maps for each subject whose task file is
    listed in taskList
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
        id = os.path.basename(tfile).split('-')[1].split('_')[0]
        confounds = get_confounds(id, motiondir, False)
        scanDur = confounds.shape[0]*2.5 #CIMAQ fMRI TR = 2.5s
        events = extract_events(tfile, id, ev_outdir, scanDur)
        get_subject_betas(id, confounds, events, fmridir, b_outdir)
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
