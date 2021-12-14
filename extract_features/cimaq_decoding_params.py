#!/usr/bin/python3

import numpy as np
from sklearn.utils import Bunch


apply_defs = Bunch(smoothing_fwhm=8,
                   dtype='f',
                   ensure_finite=False)

# To set manually: t_r,
clean_defs = Bunch(standardize=False,
                   standardize_confounds=False,
                   high_pass=None, low_pass=None,
                   ensure_finite=True)

# To set manually: mask_img
masker_defs = Bunch(t_r=None, smoothing_fwhm=None,
                    standardize=False, standardize_confounds=False,
                    high_variance_confounds=False, detrend=False,
                    low_pass=None, high_pass=None)

# To set manually: frame_times
design_defs = Bunch(drift_model=None,
                    hrf_model='spm')

# To set manually: t_r, mask_img, target_shape, target_affine
glm_defs = Bunch(drift_model=design_defs['drift_model'],
                 standardize=False,
                 smoothing_fwhm=None,
                 signal_scaling=(0, 1),  # 0, 1 or (0, 1)
                 noise_model='ar1',
                 hrf_model=design_defs['hrf_model'],
                 minimize_memory=False)

_params = Bunch(apply_defs=apply_defs,clean_defs=clean_defs,
                masker_defs=masker_defs, design_defs=design_defs,
                glm_defs=glm_defs)
