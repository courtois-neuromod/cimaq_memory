#how to import mat files into python
#https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

##Extract and import motion parameters per get_subject_score
#Create confounds_subID.tsv file for each subject

#also calculate, and save in main directory:
#average motion per param, mean motion,
#flag outliers? Worst value? Generate plot?

import os
import sys
import argparse
import glob

import numpy as np
import scipy
import pandas as pd
from numpy import nan as NaN
from scipy import io as sio

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        Input: Folder
        """)

    parser.add_argument(
        "-d", "--idir",
        required=True, nargs="+",
        help="Folder with input files",
        )

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output folder - if doesn\'t exist it will be created.")

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

#List of _extra.mat files
def get_all_mFiles(iDir):
    if not os.path.exists(iDir):
        sys.exit('This folder doesnt exist: {}'.format(iDir))
        return
    all_mFiles = glob.glob(os.path.join(iDir,'*confounds.tsv'))
    return all_mFiles

def get_id(sfile):
    filename = os.path.basename(sfile)
    subid = filename.split('_')[1]
    id = subid.split('sub')[1]
    return id

def get_subject_counfounds(mfile, o_dir):
    id = get_id(mfile)

    sfile = pd.read_csv(mfile, sep='\t')

    tot_frames = sfile.shape[0]
    tot_scrubbed = sfile['scrub'].sum(axis=0)

    motion = sfile.iloc[:, 0:7]
    meanMotion = motion.abs().mean(axis=0)

    meanWM = sfile['wm_avg'].mean(axis=0)
    meanVent = sfile['vent_avg'].mean(axis=0)

    #id number, number of frames, number of scrubbed frames
    #mean for seven motion parameters, white matter signal and ventricle signal
    means = [id, tot_frames, tot_scrubbed, meanMotion['motion_tx'], meanMotion['motion_ty'],
    meanMotion['motion_tz'], meanMotion['motion_rx'], meanMotion['motion_ry'], meanMotion['motion_rz'], meanMotion['FD'],
    meanWM, meanVent]
    return means

def extract_results(mFiles, out_dir):
    meanData = pd.DataFrame()
    colNames = ['id', 'total_frames', 'total_scrubbed', 'mean_motion_tx', 'mean_motion_ty',
    'mean_motion_tz', 'mean_motion_rx', 'mean_motion_ry', 'mean_motion_rz', 'mean_FD',
     'mean_white_matt_sig', 'mean_ventri_sig']

    for i in range(0, len(colNames)):
        meanData.insert(loc=len(meanData.columns), column=colNames[i], value=NaN,
        allow_duplicates=True)

    #id is a string, default value cannot be NaN
    meanData[['id']]=meanData[['id']].astype('object', copy=False)

    for mFile in mFiles:
        sdata = get_subject_counfounds(mFile, out_dir)
        meanData = meanData.append(pd.Series(sdata, index=meanData.columns), ignore_index=True)

    return meanData



def main():
    args = get_arguments()
    all_mFiles = get_all_mFiles(args.idir[0])
    output_dir = os.path.join(args.odir[0], 'MotionResults')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #extract_results(all_mFiles, output_dir)
    meanMotion = extract_results(all_mFiles, output_dir)
    meanMotion.to_csv(output_dir+'/fMRI_meanMotion.tsv',
    sep='\t', header=True, index=False)

if __name__ == '__main__':
    sys.exit(main())
