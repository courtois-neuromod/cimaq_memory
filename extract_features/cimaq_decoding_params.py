#!/usr/bin/python3

import numpy as np
from sklearn.utils import Bunch


apply_defs = Bunch(smoothing_fwhm=8,
                   dtype='f',
                   ensure_finite=True)

# To set manually: t_r, mask_img
clean_defs = Bunch(detrend=False, standardize=False,
                   low_pass=None, high_pass=None,
                   ensure_finite=True)

# To set manually: mask_img, t_r
masker_defs = Bunch(standardize=False,
                    standardize_confounds=False,
                    high_variance_confounds=False,
                    smoothing_fwhm=None,
                    detrend=False,
                    dtype='f',
                    low_pass=None, high_pass=None,
                    allow_overlap=True)


# To set manually: frame_times
design_defs = Bunch(drift_model=None,
                    hrf_model='spm')

# To set manually: t_r, mask_img, target_shape, target_affine
# ``signal_scaling`` is incompatible with ``standardize``
# -> Enforces ``standardize=False``
# ``slice_time_ref=0.5`` (FMRIPrep default)
# Source:
# https://fmriprep.org/en/stable/api.html?highlight=reference%20slice#fmriprep.config.workflow.slice_time_ref
glm_defs = Bunch(drift_model=None,
                 t_r=None,
                 drift_order=None,
                 mask_img=False,
                 slice_time_ref=0.5,
                 standardize=False,
                 smoothing_fwhm=None,
                 signal_scaling=0,  # 0, 1, (0, 1) or False
                 noise_model='ar1',
                 hrf_model='spm',
                 minimize_memory=False)

_params = Bunch(apply_defs=apply_defs,
                clean_defs=clean_defs,
                design_defs=design_defs,
                glm_defs=glm_defs,
                masker_defs=masker_defs)
