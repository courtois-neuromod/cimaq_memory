#!usr/bin/python3

import json
import loadutils as lu
import nibabel as nib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import nilearn
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level import FirstLevelModel
import numpy as np
from operator import itemgetter
import os
from os.path import expanduser as xpu
from os.path import join as pjoin
from os.path import splitext
import pandas as pd
from tqdm import tqdm
from make_events_df import make_events_df
from typing import Iterable
from typing import Union

dspath=os.environ.get('DATASETS')
output_dir=pjoin(dspath,'cimaq_decoding_files')
def create_bmap(sub_dir:Union[str,os.PathLike],
                noise_model:str='ar1',
                hrf_model:str='spm',
                drift_model:str=None,
                fwhm:int=8,
                **kwargs):
    sub_id = os.path.basename(sub_dir)
    tsk_prfx = '_ses-04_run-01_task-memory_'
    outfile_suffix = '_bmap-effectsize.nii.gz'
    confounds=pd.read_csv(pjoin(sub_dir,sub_id+tsk_prfx+'confounds.tsv'),sep='\t')
    events = pd.read_csv(pjoin(sub_dir,sub_id+tsk_prfx+'events.tsv'),sep='\t')
    contrast_list=[]
    sub_out_dir=pjoin(xpu(output_dir),'derivatives',sub_id,'beta_maps')
    out_filename=pjoin(sub_out_dir,sub_id+outfile_suffix)
    os.makedirs(sub_out_dir,exist_ok=True)
    fmri_img = nib.load(pjoin(sub_dir,sub_id+tsk_prfx+'bold.nii.gz'))
    nscans, t_r = fmri_img.shape[-1],fmri_img.header.get_zooms()[-1]
    frame_times = frame_times = np.arange(confounds.shape[0])*t_r
    for row in tqdm(list(events.iterrows())):
        tnum=row[1].trial_number
        events['trial_type'] = ['X_'+row[1].condition
                                if row[1].trial_number!=tnum
                                else row[1].condition
                                for row in events.iterrows()]
        mat_params = {'frame_times':frame_times,
                      'events':events[['onset', 'duration', 'trial_type']],
                      'add_regs':confounds,
                      'drift_model':drift_model,
                      'hrf_model':hrf_model}
        trial_matrix = make_first_level_design_matrix(**mat_params)
        trial_contrast = pd.Series(np.array([1]+list(
                             np.repeat(0,trial_matrix.shape[1]-1)))).values
        glm_params = {'t_r':t_r,
                      'drift_model':drift_model,
                      'standardize':True,
                      'noise_model':noise_model,
                      'hrf_model':hrf_model,
                      'smoothing_fwhm':fwhm}
        fit_params = {'run_imgs':fmri_img,'design_matrices':trial_matrix}
        con_params = {'contrast_def':trial_contrast,
                      'output_type':'effect_size'}
        contrast_list.append(
            FirstLevelModel().fit(**fit_params).compute_contrast(**con_params))
    nib.save(img=nilearn.image.concat_imgs(contrast_list),
             filename=out_filename)

def main():
    if '__name__'=='__main__':
        create_bmap(sub_id,events,fmri_img,frame_times,
                    output_dir,t_r,noise_model,hrf_model,
                    drift_model,confounds,fwhm,**kwargs)
