# cimaq_memory

### Directories' content and purpose overview

- NOTE: Each directory has a more detailed readme.txt file
        on the scripts and files they contain.

#### 1. eprime2events directory

- Contents:
   cimaq_convert_eprime_to_bids_event.py
   Converts .txt files outputed by eprime into .tsv "event"
   files compatible with Nistats/Nilearn (to label fMRI frames
   based on condition (encoding vs control task) and task performance,
   - E.g.: Subsequent recognition of image and its on-screen position
   - NOTE: Data to test the script can be found [here](https://www.dropbox.com/sh/1ijtytsqu3zghec/AAAnzupem2TA45EGyW9QpuH9a?dl=0)

#### 2. subject_details directory

- Header description .json files

    1- MainParticipantFile_headers.json
       Describes the headers of the main
       cimaq participant table
       (saved in main Participant directory
        following bids format)

    2- MemoTaskParticipantFile_headers.json

       Describes the columns from SubjectList.tsv,
       the task-specific (fMRI memory) subject data table
       (eventually saved under /cimaq_memory/Participants)

   Scripts

     1- TaskBehavAnalyses.py
        Extracts subject-specific memory performance metrics
        (e.g., hits-FA, dprime) from single-subject Post*.tsv files
        produced by cimaq_convert_eprime_to_bids_event.py
        (saved in single table with 1 row per subject).
        - NOTE: Data to test the script can be found [here](https://www.dropbox.com/sh/92rztaa42bfr5m4/AADK6ACkCf33Ri_1o_vr4xNTa?dl=0)

     2- ExtractMotionInfo.py
        Compiles a single .tsv table (for all subjects) of
        motion parameters (e.g., mean motion in 6 regressed-out directions,
        number of scrubbed frames) from *confounds.tsv files outputed
        by NIAK (found under resample directory)
        - NOTE: Data to test the script can be found [here](https://www.dropbox.com/sh/7japgtb6a64qk7n/AAB4M9e1aOFgCVD1-iI0e8cSa?dl=0)

     3- MakeMotionFiles_fromMat.py
        Creates .tsv motion files that can be regressed out of models
        in Nistats/Nilearntakes based on *_extra.mat files
        outputed by NIAK during pre-processing.
        Also generates a table of motion metrics for all participants.
        (obsolete: confounds not extracted from .mat files anymore)

#### 3. extract_features directory (WIP)

     1- cimaq_getbetas.py

        Produces brain maps of beta weights;
        each map reflects the modeling of the hemodynamic response
        (spm model; in Nistats) for a single trial.

     2- CIMAQ_getBetaMaps.ipynb

        Notebook detailing how to compute beta maps
        using a single participant as an example.

     3- CIMAQ_getNetworkFeatures.ipynb

        Notebook detailing how to compute network activation features
        using a single participant as an example.


#### 4. models directory (WIP)

     1- CIMAQ_wholeBrainSVM.ipynb

        Notebook where an SVM is trained to classify among trials
        from the cimaq task (e.g., encoding vs control task)
        using all brain voxeks
