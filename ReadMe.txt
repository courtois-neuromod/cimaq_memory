Overview of the different directories' content and purpose

NOTE: you can refer to each directory's readme.txt file for
more details on the scripts and files they contain.
The current file is an overview.

1. eprime2events directory

Contains the cimaq_convert_eprime_to_bids_event.py script
and supporting documents.
The script converts .txt files outputed by eprime into .tsv event
files compatible with Nistats/Nilearn (to label fMRI frames
based on condition (encoding vs control task) and task performance,
e.g., subsequent image and source recognition)


2. subject_details directory

Contains scripts that extract info from subject-specific files,
process them and output the data in a single table for all
participants (with 1 row per participant).

- TaskBehavAnalyses.py script: takes single-subject Post*.tsv
files produced by the cimaq_convert_eprime_to_bids_event.py script,
extracts subject-specific memory performance metrics
(e.g., hits-FA, dprime), and saves them in a single table with
1 row per subject.

- (obsolete) MakeMotionFiles_fromMat.py: takes extra.mat files created by NIAK
during pre-processing, and create .tsv motion files that can be regressed out
of models in Nistats/Nilearn. Also generates a table of motion metrics with
one line for each participant (e.g., mean motion along 6 axes,
total number of scrubbed frames)

- ExtractMotionInfo.py: ...

- MemoTaskParticipantFile_headers.json: descriptions of the
columns in SubjectList.tsv, the task-specific subject info table
(eventually saved under /cimaq_memory/Participants)

And it contains the MainParticipantFile_headers.json file that
describes the headers of the whole cimaq participant table
(saved in main Participant directory following bids format)
