#!/usr/bin/env python
# encoding: utf-8

#1.
##Create events.tsv files to create nistats design matrices
#Nistats: how to make events.tsv file to extract 1 beta map per trial
#https://nistats.github.io/auto_examples/04_low_level_functions/write_events_file.html#sphx-glr-auto-examples-04-low-level-functions-write-events-file-py

#2. Load confounds.tsv file (to regress out motion, bad frames, etc)

#3. Create a design matrix (model)
#https://nistats.github.io/auto_examples/04_low_level_functions/plot_design_matrix.html#sphx-glr-auto-examples-04-low-level-functions-plot-design-matrix-py

#4. Fit the model onto data

#5. Use regression model to create beta maps for each

#6. get group mask from niak preprocessing

#7. vectorize data, or compute correlations betweeen b maps and connectome r maps (features)

import os
import sys
import argparse
import glob
import re
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
        Convert behavioural data from cimaq to bids format
        Input: Folder
        """)

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
        help="Folder with motion files",
        )

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output    folder - if doesnt exist it will be created",
        )

    parser.add_argument(
        "-v", "--verbose",
        required=False, nargs="+",
        help="Verbose to    get more information about what's going on",
        )

    args =  parser.parse_args()
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def get_all_Files(inDir):
    if not os.path.exists(inDir):
        sys.exit('This folder does not exist: {}'.format(inDir))
        return
    all_Files = glob.glob(os.path.join(inDir,'*events.tsv'))
    return all_Files

def get_confounds(id, motiondir):
    mfile = glob.glob(os.path.join(motiondir, 'fmri_sub'+id+'*confounds.tsv'))
    if len(mfile) == 0:
        print('Motion file missing for participant '+id)
        return
    else:
        smf = pd.read_csv(mfile[0], sep='\t')

        short_smf = smf.copy(deep=True)

        colsKeep = ['motion_tx', 'motion_ty', 'motion_tz', 'motion_rx',
        'motion_ry', 'motion_rz', 'scrub', 'wm_avg', 'vent_avg']

        cols = smf.columns

        for i in cols:
            if (i in colsKeep)==False:
                short_smf.drop([i], axis=1, inplace=True)

    #decide which confounds to use in model: short or full (smf) list
    #return short_smf
    return smf

#load task file, add columns specific to my analysis
#create events dataframe fed to first-level model
def extract_sub(subFile, id, outdir, scanDur):
    stf = pd.read_csv(subFile, sep='\t')

    #add column with number that differentiates each encoding trial
    #so that each trial is modelled separately (gets own map of beta values)
    #same number used for control trials (modelled but not analysed)
    numCol = stf.shape[1]
    insertCol = [3, 4, numCol+2, numCol+3]
    colNames = ['trial_idx', 'trial_label', 'unscanned', 'enctrial_type']
    colVals = [-1, 'None', 0, 'None']
    for i in range(0, len(colNames)):
        stf.insert(loc=insertCol[i], column=colNames[i], value=colVals[i], allow_duplicates=True)

    #Fill vector columns that identify missed, wrong source and correct source trials
    count = 1
    for i in stf[stf['trial_type']=='Enc'].index:
        stf.loc[i, 'trial_idx'] = count
        stf.loc[i, 'trial_label'] = 'Enc'+str(count)
        if stf.loc[i, 'position_accuracy'] == 0:
            stf.loc[i, 'enctrial_type']='missed'
        elif stf.loc[i, 'position_accuracy'] == 1:
            stf.loc[i, 'enctrial_type']='wrongsource'
        elif stf.loc[i, 'position_accuracy'] == 2:
            stf.loc[i, 'enctrial_type']='correctsource'
        count = count + 1

    #control trials modelled as single condition
    for j in stf[stf['trial_type']=='CTL'].index:
        stf.loc[j, 'trial_idx'] = count
        stf.loc[j, 'trial_label'] = 'Control'

    #flag trials for which no brain data was acquired
    for k in stf[stf['offset']>scanDur].index:
        stf.loc[k, 'unscanned']=1

    #events files output subdirectory
    outdir_ev = os.path.join(outdir, 'Events')
    if not os.path.exists(outdir_ev):
        os.mkdir(outdir_ev)

    #save extended task file dataframe as tsv file
    stf.to_csv(outdir_ev+'/sub-'+id+'_events.tsv',
    sep='\t', header=True, index=False)

    #remove trials for which there is no corresponding fMRI data
    #because scan cut short
    stf = stf[stf['unscanned']==0]

    #Save vector of conditions per trial (encoding vs control)
    #to label trials for classification in later analyses
    ttypes = stf['trial_type']
    ttypes.to_csv(outdir_ev+'/sub-'+id+'_AllTrialTypes.tsv',
    sep='\t', header=True, index=False)

    #Save vector of encoding trial types (missed) to label encoding trials
    #according to subsequent memory performance in later analyses
    encTrials = stf[stf['trial_type']=='Enc']
    vecs = encTrials.iloc[:, (encTrials.shape[1]-1):]
    vecs.to_csv(outdir_ev+'/sub-'+id+'_EncTrialTypes.tsv',
    sep='\t', header=True, index=False)

    return stf

def onemodel_ev(s_events):
    #col1 = onsets, 2 = duration, 3 = trial_idx
    events = s_events.iloc[:, 1:4]
    events.rename(columns={'trial_idx':'trial_type'}, inplace=True)
    return events

def multimodels_ev(s_events):
    count = s_events[s_events['trial_type']=='Enc'].shape[0] + 1
    for m in s_events[s_events['trial_type']=='CTL'].index:
        s_events.loc[m, 'trial_idx'] = count
        count = count+1

    events = s_events.iloc[:, 1:4]
    events.rename(columns={'trial_idx':'trial_type'}, inplace=True)
    return events

def get_betas(id, s_model, im_outdir, numEnc):
    design_matrix = s_model.design_matrices_[0]
    #numpy eye: return a 2-D array w ones on diagonal and zeroes elsewhere
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
    contrast_matrix = np.eye(design_matrix.shape[1]) #dimensions = num columns

    #dictionnary of constrasts : weight of 1 for corresponding design matrix column, 0 elsewhere
    contrasts = dict([(column, contrast_matrix[i])
    for i, column in enumerate(design_matrix.columns[0:numEnc])])

    #Compile ordered list of beta maps and their trial number
    enc_betas_filelist = []

    #compute the per-contrast beta maps,
    #export b_map .nii image in output directory
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        b_map = s_model.compute_contrast(contrast_val, output_type='effect_size') #for betas
        b_name = os.path.join(im_outdir, 'betas_sub'+str(id)+'_Enc'+str(contrast_id)+'.nii')
        nibabel.save(b_map, b_name)
        print(os.path.basename(b_name))
        enc_betas_filelist.append(b_name)

    allenc_betas = nibabel.funcs.concat_images(images=enc_betas_filelist, check_affines=True, axis=None)

    print(allenc_betas.shape)
    nibabel.save(allenc_betas, os.path.join(im_outdir, 'concat_enc_betas_sub'+str(id)+'.nii'))

    return

#nistats.first_level_model is interface to use the glm implemented in nistats.regression
def run_glm(id, confounds, OM_events, MM_events, fmridir, outdir, numEnc):
    #set up output directory structure
    im_outdir = os.path.join(outdir, str(id))
    if not os.path.exists(im_outdir):
        os.mkdir(im_outdir)
    im_outdir_OM = os.path.join(im_outdir, 'SingleModel')
    im_outdir_MM = os.path.join(im_outdir, 'MultiModels')
    os.mkdir(im_outdir_OM)
    os.mkdir(im_outdir_MM)

    scanfile = glob.glob(os.path.join(fmridir, 'fmri_sub'+id+'*nii'))
    if len(scanfile) == 0:
        print('fMRI data file missing for participant '+id)
        return
    else:
        fmri_imgNS = scanfile[0]
        #8mm smoothing since using unsmoothed preprocessed data (NIAK output: resample)
        fmri_img = image.smooth_img(fmri_imgNS, 8)
        tr = 2.5  #CIMAQ fMRI = 2.5s TRs
        n_scans = confounds.shape[0] #number of frames
        frame_times = np.arange(n_scans)*tr # corresponding frame times
        hrf_model = 'spm' #or 'glover' or 'spm + derivative'

        #Create design matrix; slow drift modelled with NIAK preprocessing pipeline
        design = make_first_level_design_matrix(frame_times, events=OM_events,
        drift_model=None, add_regs=confounds, hrf_model=hrf_model)

        s_model = FirstLevelModel(t_r=tr, drift_model = None, standardize = True,
        noise_model='ar1', hrf_model = hrf_model)

        s_model = s_model.fit(fmri_img, design_matrices = design)

        get_betas(id, s_model, im_outdir_OM, numEnc)

        #compile list of beta maps (all trials) to concatenate them in a 4D file
        all_betas_filelist=[]
        numTrials = MM_events.shape[0]
        for i in range(0, numTrials):
            #make copy of orig_events that does not modify the original
            events = MM_events.copy(deep=True)

            #condition number for that trial
            tnum = events.iloc[i, 2]
            if tnum < (numEnc+1):
                currentCondi = 'Enc'
            else:
                currentCondi = 'CTL'

            #modify trial_type column to model only the trial of interest (tnum)
            for j in events.index:
                if events.loc[j, 'trial_type'] == tnum:
                    events.loc[j, 'trial_type']= currentCondi+str(tnum)
                elif events.loc[j, 'trial_type'] < (numEnc+1):
                    events.loc[j, 'trial_type']='X_Enc' #X for condition alphabetical order
                else:
                    events.loc[j, 'trial_type']='X_CTL' #X for alphabetical order

            #create the design matrix
            design = make_first_level_design_matrix(frame_times, events=events,
            drift_model=None, add_regs=confounds, hrf_model=hrf_model)

            #define first level model parameters
            s_model = FirstLevelModel(t_r=tr, drift_model = None, standardize = True,
            noise_model='ar1', hrf_model = hrf_model)

            #fit model with desigm matrix
            s_model = s_model.fit(fmri_img, design_matrices = design)

            #get contrasts
            design_matrix = s_model.design_matrices_[0]
            contrast_vec = np.repeat(0, design_matrix.shape[1])
            contrast_vec[0] = 1

            if tnum == 32:
                contrast_EncMinusCTL = np.repeat(0, design_matrix.shape[1])
                contrast_EncMinusCTL[1] = -1
                contrast_EncMinusCTL[2] = 1
                b_tasks = s_model.compute_contrast(contrast_EncMinusCTL, output_type='effect_size') #for betas
                t_tasks = s_model.compute_contrast(contrast_EncMinusCTL, stat_type='t', output_type='stat') #for tscores
                nibabel.save(b_tasks, os.path.join(im_outdir_MM, 'EncMinCTL_betas_sub'+str(id)+'.nii'))
                nibabel.save(t_tasks, os.path.join(im_outdir_MM, 'EncMinCTL_tscores_sub'+str(id)+'.nii'))

            b_map = s_model.compute_contrast(contrast_vec, output_type='effect_size') #for betas
            b_name = os.path.join(im_outdir_MM, 'betas_sub'+str(id)+'_Trial'+str(i+1)+'_'+currentCondi+str(tnum)+'.nii')
            print(os.path.basename(b_name))
            all_betas_filelist.append(b_name)
            nibabel.save(b_map, b_name)

        alltrials_betas = nibabel.funcs.concat_images(images=all_betas_filelist, check_affines=True, axis=None)
        print(alltrials_betas.shape)
        nibabel.save(alltrials_betas, os.path.join(im_outdir_MM, 'concat_all_betas_sub'+str(id)+'.nii'))

    return

def extract_files(fileList, outdir, motiondir, fmridir):
    for file in fileList:
        id = os.path.basename(file).split('-')[1].split('_')[0]
        s_confounds = get_confounds(id, motiondir)
        scanDur = s_confounds.shape[0]*2.5 #TR = 2.5s
        s_events = extract_sub(file, id, outdir, scanDur)
        numEnc = s_events[s_events['trial_type']=='Enc'].shape[0]
        OM_events = onemodel_ev(s_events) #one model for all trials
        MM_events = multimodels_ev(s_events) #one model per trial
        run_glm(id, s_confounds, OM_events, MM_events, fmridir, outdir, numEnc)
    return

def main():
    args =  get_arguments()
    all_Files = get_all_Files(args.tdir[0])
    m_dir = args.mdir[0]
    f_dir = args.fdir[0]
    out_dir = os.path.join(args.odir[0], 'Output')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    extract_files(all_Files, out_dir, m_dir, f_dir)

if __name__ == '__main__':
    sys.exit(main())
