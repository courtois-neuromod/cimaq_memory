Overview of the different directories' content and purpose

NOTE: you can refer to each directory's readme.txt file for
more details on the scripts and files they contain.

1. eprime2events directory

Contains the cimaq_convert_eprime_to_bids_event.py script
and supporting documents.
The script converts .txt files outputed by eprime into .tsv event
files compatible with Nistats/Nilearn (to label fMRI frames
based on condition (encoding vs control task) and task performance,
e.g., subsequent image and source recognition)


2. subject_details directory

Contains header description .json files:
- MainParticipantFile_headers.json describes the headers
of the main cimaq participant table
(saved in main Participant directory following bids format)

- MemoTaskParticipantFile_headers.json describes the
columns from SubjectList.tsv, the task-specific (fMRI memory)
subject data table (eventually saved under /cimaq_memory/Participants)

Also contains 3 scripts:
- TaskBehavAnalyses.py script: extracts subject-specific
memory performance metrics (e.g., hits-FA, dprime) from single-subject
Post*.tsv files produced by the cimaq_convert_eprime_to_bids_event.py
script (save in single table with 1 row per subject).

- MakeMotionFiles_fromMat.py: creates .tsv motion files that can be
regressed out of models in Nistats/Nilearntakes based on *_extra.mat files
outputed by NIAK during pre-processing.
Also generates a table of motion metrics for all participants.
(obsolete: confounds not extracted from .mat files anymore)

- ExtractMotionInfo.py: compiles a single .tsv table (all subjects) of
motion parameters (e.g., mean motion in 6 regressed-out directions,
number of scrubbed frames) from *confounds.tsv files outputed
by NIAK (found under resample directory)


3. extract_features directory

- cimaq_getbetas.py script: produces brain maps of beta weights;
each map reflects the modeling of the hemodynamic response
(spm model; in Nistats) for a single trial.

- CIMAQ_getBetaMaps.ipynb: jupyter notebook detailing how beta maps
are computer in a single participant

- CIMAQ_getNetworkFeatures.ipynb: jupyter notebook detailing how
network activation features are computed in a single participant


4. models directory

- CIMAQ_wholeBrainSVM.ipynb: jupyter notebook where an SVM is trained
to classify among trials from the cimaq task (e.g., encoding vs control task)
using all brain voxeks
