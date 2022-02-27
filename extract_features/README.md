# Trial-Unique Beta Maps Creation Processing Steps

## From the CIMAQ Memory Task (Image Encoding) fMRI Data - Within-Subject Level

### Goal: Feed Outputs (Beta Maps) as Features to a Within-Subject Nilearn Classifier

##### Input:

- Event files
- Confounds (motion, etc) files
    - Generated by load_confound
- Preprocessed FMRIPrep data (4D .nii file)

**- Note: Data NOT Smoothed nor Denoised

##### Output:

- 1 map (3D .nii file) of beta (regression) weights for each trial
- 1 concatenated 4D file of these 3D maps (trials ordered chronologically).

#### Version 1: Separate Model for Each Trial

- Trial of interest modelled as a separate condition (1 regressor)
- All other trials modelled in either the Encoding or Control condition (2 regressors)

Reference: How to derive beta maps for MVPA classification (Mumford et al., 2012):

https://www.sciencedirect.com/science/article/pii/S1053811911010081

#### Also creating contrasts per condition (to derive features for between-subject classification):

 - Modeling enconding and control conditions across trials
     - 3 beta maps:
         - encoding (enc) , control (ctl), and encoding minus control (enc_minus_ctl)
 - Modeling control condition, as well as the encoding condition according to task performance:
    - miss and hit (post-scan image recognition performance)
    - 5 beta maps:
        - miss, hit hit_minus_miss, hit_minus_ctl, miss_minus_ctl
    - Modeling control condition & encoding condition according to task performance:
        - miss, wrong source, and correct source
    - 7 beta maps:
        - wrong_source, corr_source, cs_minus_ws, cs_minus_miss, ws_minus_miss, cs_minus_ctl, ws_minus_ctl


### Step 1: Load confound parameters

```
from load_confounds import Minimal

confounds = Minimal().load(FMRIPrep/preprocessed/fmri_img/file/path)
```

### From the load_confounds README document:

#### Note on low pass filtering

Common operation in resting-state fMRI analysis
- Featured in all preprocessing strategies of the Ciric et al. (2017) paper

fMRIprep does not output low pass filtering discrete cosines
- Can be implemented directly with the nilearn masker
    - ``low_pass`` argument
**Specify the nilearn masker argument ``t_r`` if low_pass is used

#### Note on high pass filtering and detrending

Nilearn maskers & first-level model can remove slow time drifts & noise:
- ``high_pass`` & ``detrend`` arguments
- Both **redundant** with fMRIprep high_pass regressors
    - Both included in all load_confounds strategies
**Do NOT use nilearn's ``high_pass`` or ``detrend`` options with the default strategies.**

- A flexible ``Confounds`` loader can exclude fMRIprep high_pass noise components
    - Allows relying on nilearn's ``high_pass`` or ``detrending`` options
    **- NOT advised with compcor or ica_aroma analysis**

#### Note on demeaning confounds

**Confounds should be demeaned** (default load_confounds behaviour)
- Required to properly regress out confounds using nilearn
    - With the standardize=False, standardize=True or standardize="zscore" options
    - standardize="psc" requires turning off load_confounds demeaning option
    ```
    from load_confounds import Params6
    conf = Params6(demean=False)
    ```
    - Unless using nilearn maskers or first-level model ``detrend`` or ``high_pass`` options


### Step 2: create events variable & events.tsv file

#### From the 'sub-*_ses-V*_task-memory_events.tsv' file outputed by cimaq2bids.py

Number of rows = number of trials

- First-level model uses trial onset times to match trial conditions to fMRI frames

Documentation:

https://nistats.github.io/auto_examples/04_low_level_functions/write_events_file.html#sphx-glr-auto-examples-04-low-level-functions-write-events-file-py

- Each encoding trial is modelled as a different condition (under trial_type column)
    - Modelled separately in the design matrix
        - Trial of interest has its own column in the design matrix
        - Other columns = other trials &  confound regressors
            - Modelled together as a single regressor

**Note: Some scans were cut short**
- The last few trials have NO associated brain activation frames
    - These need to be left out of the analysis
- MEMO: 310 frames = full scan, 288 frames = incomplete (~15 participants).

- "unscanned" trials need to be excluded from the model (about ~2-4 trials missing).

- E.g.:
    - 288*2.5 = 720s.
    - Trial #115 (out of 117) offset time ~ 710s
    - Trial #116 (out of 117) onset ~ 723s


### Step3 : Implement first-level model (implements regression in nilearn)

#### Generates contrasts and output maps of beta values (parameter estimators; one map of betas per trial).

**About first-level model:

Note 1: ``nilearn.glm.first_level_model`` provides an interface for ``nilearn.glm``

Note 2: Each encoding trial is modelled as a separate condition to obtain separate maps of beta values
- Model's output type to get betas = **effect sizes**

    - ``nilearn.glm.first_level.FisrtLevelModel.compute_contrast`` ``output_type`` parameter name

- Version A:
    - Control trials & encoding trials are modelled separately
        - 2 regressors
- Version B:
    - Control trials & encoding trials are modelled together
        - Single "other_trials" condition (1 regressor)

Note 3: the first_level_model can either be given its parameters in 2 ways:

1. Pre-constructed **design matrix**
    - 2-step method (chosen method here)
    - Built from the events and confounds files in a separate preparatory step
    - Takes precedence over the events and confounds parameters (method 2)
2. Events & confounds directly
    - 1-step method
    - Skipping the need to create a design matrix in a separate step
    - The model will generate the design matrix automatically


Nilearn links on design matrices:

https://nilearn.github.io/modules/generated/nilearn.plotting.plot_design_matrix.html#nilearn.plotting.plot_design_matrix

https://nilearn.github.io/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html#nilearn.glm.first_level.make_first_level_design_matrix

##### Examples

- First-Level Model

https://nilearn.github.io/modules/generated/nilearn.glm.first_level.FirstLevelModel.html#nilearn.glm.first_level.FirstLevelModel

- Beta Map Extraction (Nistats)

https://github.com/poldracklab/fitlins/pull/48

**About contrasts and maps of beta values (parameter estimators):

- To access the estimated coefficients (betas of the GLM model)
    - We need to specify "canonical contrasts" (one per trial) isolating design matrix columns
        - Each contrast has a single 1 in its corresponding colum, and 0s for all the other columns