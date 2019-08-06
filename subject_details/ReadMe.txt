
Overview of the subject-details directory content:

- 3 python scripts:
- 2 .json files describing the headers of data tables

MemoTaskParticipantFile_headers.json :
describes the metrics found in each column of the SubjectList.tsv,
the task-specific subject info table (one line per participant)
(eventually saved under /cimaq_memory/Participants).

MainParticipantFile_headers.json :
describes the headers of the cimaq participant data table
(eventually saved in main Participant directory following bids format)

The TaskBehavAnalyses.py script :
Input:
 -d: path to directory that contains a series of
 Post*.tsv files (outputed by the cimaq_convert_eprime_to_bids_event.py
 script, will be under /cimaq_memory/TaskFiles/Processed)
 -o: path to output directory where output files will be saved
(eventually /cimaq_memory/Participants)

Note: three Post*.tsv files are included in Dropbox to test the script

Output:
- a single .tsv file that contains performance metrics on the post-scan
memory recognition task (e.g., hits-FA, dprime).
Saved under /cimaq_memory/Participants/TaskResults),
with one row per participant.


The ExtractMotionInfo.py script:

Takes confounds.tsv files outputed by NIAK (resample directory), extract values, and
compiles a single table for all subjects

Input:
-d
-o

Output:

The MakeMotionFiles_fromMat.py script:
takes extra.mat files created by NIAK
during pre-processing, and create .tsv motion files that can be regressed out
of models in Nistats/Nilearn.
Also generates a table of motion metrics with
one line for each participant (e.g., mean motion along 6 axes,
total number of scrubbed frames)
