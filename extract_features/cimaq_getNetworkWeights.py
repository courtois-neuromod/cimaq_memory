#!/usr/bin/env python
# encoding: utf-8
"""
Overview:
Create one series of network weights per trial using beta maps
produced with Nistats

Note: using MIST parcellation
references: https://mniopenresearch.org/articles/1-3/v2#ref-50 https://figshare.com/articles/MIST_A_multi-resolution_parcellation_of_functional_networks/5633638/1
https://simexp.github.io/multiscale_dashboard/index.html

1. step 1

2. step 2

3. step 3
"""

import os
import sys

import glob
import scipy
import nibabel
import nilearn
import numpy as np
import pandas as pd

from numpy import nan as NaN
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from nilearn import image
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_roi, plot_anat, plot_img, show
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMasker
from nilearn.connectome import ConnectivityMeasure

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Create trial-unique brain maps of beta coefficients
        Input: Folders with task file, confounds and fMRI data
        """)

    parser.add_argument(
        "-b", "--bdir",
        required=True, nargs="+",
        help="Folder with beta map files",
        )

    parser.add_argument(
        "-m", "--mdir",
        required=True, nargs="+",
        help="Folder with MIST parcellation templates",
        )

    parser.add_argument(
        "-p", "--plevel",
        required=True, nargs="+",
        help="MIST parcellation level: 7, 12, 20, 36, 64, 122, 197, 325, 444",
        )

    parser.add_argument(
        "-c", "--cdir",
        required=True, nargs="+",
        help="Folder with confound files",
        )

    parser.add_argument(
        "-k", "--kdir",
        required=True, nargs="+",
        help="Folder with functional mask files",
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

    args =  parser.parse_args()
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def get_fmri_files(fDir):
    """ Returns a list of beta files (one file per participant)
    Parameter:
    ----------
    task_files: list of strings (paths to task files)

    Return:
    ----------
    None (beta maps are saved directly in outdir)
    """
    if not os.path.exists(fDir):
        sys.exit('This folder does not exist: {}'.format(tDir))
        return
    fmri_files = glob.glob(os.path.join(fDir,'fmri_sub*.nii'))
    return fmri_files

def get_netWeigts(fmri_list, betas_dir, basc, confound_dir, mask_dir, output_dir):
    for f_file in fmri_list:
        id = os.path.basename(f_file).split('_')[1].split('sub')[1]
        betas = get_betas(id, betas_dir) #nilearn  4D image file
    return

def main():
    args =  get_arguments()
    fmri_list = get_fmri_files(args.fdir[0])
    betas_dir = args.bdir[0]
    basc = os.path.join(args.mdir[0], 'MIST_'+args.plevel[0]+'.nii')
    confound_dir = args.cdir[0]
    mask_dir = args.kdir[0]
    output_dir = os.path.join(args.odir[0], 'Network_Weights')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    get_netWeigts(fmri_list, betas_dir, basc, confound_dir, mask_dir, output_dir)

if __name__ == '__main__':
    sys.exit(main())
