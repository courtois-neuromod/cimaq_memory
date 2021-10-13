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
    all_mFiles = glob.glob(os.path.join(iDir,'*mat'))
    return all_mFiles

def get_id(sfile):
    filename = os.path.basename(sfile)
    id = filename.split('_')[1]
    return id

def get_subject_counfounds(mfile, o_dir):
    id = get_id(mfile)
    sfile = sio.loadmat(mfile)
    labels = sfile['labels_confounds']
    conf = np.transpose(sfile['confounds'])
    tframes = sfile['time_frames']
    scrub = sfile['mask_scrubbing']

    sData = pd.DataFrame()
    sData.insert(loc=len(sData.columns), column = 'time_frames', value= tframes[0])
    scrubFrames = []
    for i in range(0, scrub.shape[0]):
        scrubFrames.append(scrub[i][0])

    sData.insert(loc=len(sData.columns), column = 'mask_scrubbing', value= scrubFrames)
    numscrub = sData['mask_scrubbing'].sum(axis=0)

    for i in range(0, labels.shape[0]):
        sData.insert(loc=len(sData.columns), column = labels[i][0][0]+'_'+str(i), value= conf[i])

    nCol = sData.shape[1]

    sMotion = sData.iloc[:,(nCol-9):(nCol-2)]
    meanMotion = sMotion.abs().mean(axis=0)

    meanWM = sData[sData.columns[nCol-2]].mean()
    meanVent = sData[sData.columns[nCol-1]].mean()

    sData.to_csv(o_dir+'/confounds_'+id+'.tsv', sep='\t', header=True, index=False)
    sMotion.to_csv(o_dir+'/motion_'+id+'.tsv', sep='\t', header=False, index=False)

    #id number, number of frames, number of scrubbed frames
    #mean for seven motion parameters, white matter signal and ventricle signal
    means = [id.split('sub')[1], sData.shape[0], numscrub, meanMotion[0], meanMotion[1],
    meanMotion[2], meanMotion[3], meanMotion[4], meanMotion[5], meanMotion[6],
    meanWM, meanVent]
    return means

def extract_results(mFiles, out_dir):
    meanData = pd.DataFrame()
    colNames = ['id', 'num_TRs', 'total_scrubbed', 'mean_motion1', 'mean_motion2',
    'mean_motion3', 'mean_motion4', 'mean_motion5', 'mean_motion6', 'mean_motion7',
     'mean_whiteMattSig', 'mean_ventriSig']

    for i in range(0, len(colNames)):
        meanData.insert(loc=len(meanData.columns), column=colNames[i], value=NaN,
        allow_duplicates=True)

    #id is a string, cannot be NaN
    meanData[['id']]=meanData[['id']].astype('object', copy=False)

    for mFile in mFiles:
        sdata = get_subject_counfounds(mFile, out_dir)
        meanData = meanData.append(pd.Series(sdata, index=meanData.columns), ignore_index=True)

    return meanData



def main():
    args = get_arguments()
    all_mFiles = get_all_mFiles(args.idir[0])
    output_dir = os.path.join(args.idir[0], 'MotionFiles')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #extract_results(all_mFiles, output_dir)
    meanMotion = extract_results(all_mFiles, output_dir)
    meanMotion.to_csv(output_dir+'/fMRI_meanMotion.tsv',
    sep='\t', header=True, index=False)

if __name__ == '__main__':
    sys.exit(main())
