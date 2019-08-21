
Overview of the subject-details directory content:

- 3 python scripts
- 2 .json files describing the headers of data tables (to be saved on Stark)


MemoTaskParticipantFile_headers.json :
describes the data found in each column of the SubjectList.tsv,
the subject data table specific to the cimaq fMRI memory task
(one line per participant)
(eventually saved under /cimaq_memory/Participants).


MainParticipantFile_headers.json :
describes the headers of the cimaq participant data table
(eventually saved in main /Participant directory according to bids format)


The TaskBehavAnalyses.py script :
Input:
 -d: path to directory that contains a series of
 Post*.tsv files (outputed by the cimaq_convert_eprime_to_bids_event.py
 script, will be under /cimaq_memory/TaskFiles/Processed)
 -o: path to output directory where output files will be saved
(eventually /cimaq_memory/Participants)

Note: three Post*.tsv files are included in Dropbox to test the script
https://www.dropbox.com/sh/92rztaa42bfr5m4/AADK6ACkCf33Ri_1o_vr4xNTa?dl=0

Output:
- a single .tsv file that contains performance metrics on the post-scan
memory recognition task (e.g., hits-FA, dprime).
Saved under /cimaq_memory/Participants/TaskResults),
with one row per participant.


The ExtractMotionInfo.py script:
Input:
- d: path to directory that contains the *confounds.tsv outputed by
NIAK during preprocessing (found in resample directory)
- o: path to output directory where output file is saved

Note: three *confounds.tsv are included in Dropbox to test the script
https://www.dropbox.com/sh/7japgtb6a64qk7n/AAB4M9e1aOFgCVD1-iI0e8cSa?dl=0

Output:
- fMRI_meanMotion.tsv, a single table for all participants that contains motion metrics
(mean motion in all 6 regressed-out coordinates, number of scrubbed frames)


- The MakeMotionFiles_fromMat.py script (not used: got confounds from .tsv instead):
takes extra.mat files created by NIAK
during pre-processing, and create .tsv motion files that can be regressed out
of models in Nistats/Nilearn.
Also generates a table of motion metrics with
one line for each participant (e.g., mean motion along 6 axes,
total number of scrubbed frames)
