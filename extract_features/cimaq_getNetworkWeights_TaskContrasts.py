#!/usr/bin/env python
# encoding: utf-8
"""
Overview:
Create one series of network weights per trial using beta maps
produced with Nistats, for each of the following MIST parcellations:
7, 20 and 44 networks

Note: using MIST parcellation
references: https://mniopenresearch.org/articles/1-3/v2#ref-50 https://figshare.com/articles/MIST_A_multi-resolution_parcellation_of_functional_networks/5633638/1
https://simexp.github.io/multiscale_dashboard/index.html

1. Load files (fMRI, confounds, beta maps, function fmri mask, MIST parcellation)

2. Vectorize memory task epi scans (310 fMRI frames).
- Regress out confounds (.tsv file) computed with NIAK preprocessing pipeline
from raw fMRI signal.
- Mask data with normalized functional mri mask (non-linear)

3. Extract signal time series averaged within each of the MIST 20 networks
over the 310 fMRI frames.
- Regress out confounds computed with NIAK preprocessing pipeline from raw fMRI signal

4. Calculate a correlation matrix between the data matrices obtained in steps 2 and 3,
respectively. The result is a network per voxel data matrix (e.g., 20 x 69924) that
quantifies to what extent a voxel's signal correlates with each of the 20 networks
over each fMRI frame.

5. Vectorize beta files using the functional mask.
Result is a trial * voxels data matrix (e.g., 117 * 69924)

6. Correlate matrices from steps 4 and 5 to obtain a trial per network
correlation matrix (e.g., 117 x 20)
Each row is a set of features (event) to feed an SVM model.
"""

import os
import sys
import argparse

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
        "-s", "--slist",
        required=True, nargs="+",
        help="Path to sub_list.tsv, a list of of subject ids",
        )

    parser.add_argument(
        "-b", "--bdir",
        required=True, nargs="+",
        help="Folder with beta map directories (one per subject)",
        )

    parser.add_argument(
        "-m", "--mdir",
        required=True, nargs="+",
        help="Folder with MIST parcellation templates",
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

def get_netWeigts(ids, fmri_dir, betas_dir, basc_dir, confound_dir, mask_dir, output_dir):
    """ Outputs network weights (one per trial per participant) for
    each MIST parcellation : 7, 20 and 444 networks
        pass
    Parameter:
    ----------
    ids: list of integers (subject ids)
    fmri_dir: a string (path to directory with fmri.nii files)
    betas_dir: a strings (path to directory with beta.nii files)
    basc_dir : a strings (path to directory with MIST parcelation masks)
    confound_dir: a strings (path to directory with confound.tsv files)
    mask_dir: a strings (path to directory with functional masks)
    output_dir: a string (path to directory where to save network weights)

    Return:
    ----------
    None (network features saved directly in output directory)
    """
    # get list of parcellation maps (three parcellation grains)
    basc = []
    basc.append(os.path.join(basc_dir, 'Parcellations/MIST_7.nii'))
    basc.append(os.path.join(basc_dir, 'Parcellations/MIST_20.nii'))
    basc.append(os.path.join(basc_dir, 'Parcellations/MIST_64.nii'))
    print(basc)

    # for each level of parcellation
    for i in range (0, len(basc)):
        basc_img = image.load_img(basc[i])
        basc_name = os.path.basename(basc[i]).split('.nii')[0]

        basc_out = os.path.join(output_dir, basc_name)
        if not os.path.exists(basc_out):
            os.mkdir(basc_out)

        b_labels = os.path.join(basc_dir, 'Parcel_Information', basc_name+'.csv')
        basc_labels = pd.read_csv(b_labels, sep=';')

        # for each subject on list of dccids
        for id in ids:
            # load brain maps of beta weights (one per trial: concatenate maps in temporal order)
            b_files = os.path.join(betas_dir, 'betas*nii')
            b_files = os.path.join(betas_dir, str(id), 'TrialContrasts/betas*nii')
            betas = image.load_img(img=b_files, wildcards=True)

            print(betas.header.get_data_shape())
            print(betas.header.get_zooms())

            # load fmri data (single 4D file)
            fmri = glob.glob(os.path.join(fmri_dir, 'fmri_sub'+str(id)+'*nii'))[0]

            # load confounds (.tsv file) to regress out when extracting signal from raw epis
            confounds = glob.glob(os.path.join(confound_dir, 'fmri_sub'+str(id)+'*confounds.tsv'))[0]

            # load functional mri mask (one per subject)
            mask = image.load_img(os.path.join(mask_dir, 'func_sub'+str(id)+'_mask_stereonl.nii'))

            # use NiftiMasker object to convert epi images into a frames x voxels data matrix (310 x number of voxels)
            masker = NiftiMasker(mask_img=mask, standardize=True)
            epi_sig = masker.fit_transform(fmri, confounds = confounds)
            print(epi_sig.shape)

            # Use NiftiLabelsMasker object to vectorize each network's time series
            # into a frames x networks data matrix (e.g., 310 x 20 for MIST_20)
            label_masker = NiftiLabelsMasker(labels_img=basc_img, standardize=True,
            mask_img=mask, memory = 'nilearn_cache', verbose=0)
            roi_sig = label_masker.fit_transform(fmri, confounds = confounds)

            # Calculate a network x voxels correlation matrix between each network
            # and each voxel's activity over functional frames
            numnets = int(basc_name.split('MIST_')[1])
            vox_correls = np.empty([numnets, epi_sig.shape[1]])

            for i in range(0, numnets):
                for j in range(0, epi_sig.shape[1]):
                    vox_correls[i, j] = pearsonr(epi_sig[:, j], roi_sig[:, i])[0]

            # Vectorize beta maps into a trial * voxels data matrix
            beta_mat = masker.fit_transform(betas)

            # Compute a trial per network correlation matrix (e.g., 117 x 20)
            features = np.empty([beta_mat.shape[0], numnets])
            for i in range(0, beta_mat.shape[0]):
                for j in range(0, numnets):
                    features[i, j] = pearsonr(beta_mat[i, :], vox_correls[j, :])[0]

            data = pd.DataFrame(data=features, columns = basc_labels['label'])

            data.to_csv(basc_out+'/sub-'+str(id)+'-'+basc_name+'networks_117Trials.tsv',
            sep='\t', header=True, index=False)

    return

def main():
    args =  get_arguments()
    # sub_list.tsv, a list of subject dccids in .tsv format
    slist = pd.read_csv(args.slist[0], sep = '\t')
    slist.set_index('sub_ids', inplace=True)
    ids = slist.index

    fmri_dir = args.fdir[0]
    beta_dir = args.bdir[0]
    basc_dir = args.mdir[0]
    confound_dir = args.cdir[0]
    mask_dir = args.kdir[0]
    output_dir = args.odir[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    get_netWeigts(ids, fmri_dir, beta_dir, basc_dir, confound_dir,
    mask_dir, output_dir)

if __name__ == '__main__':
    sys.exit(main())
