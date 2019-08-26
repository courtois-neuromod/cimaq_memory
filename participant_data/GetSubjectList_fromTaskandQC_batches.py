import os
import sys
import argparse
import glob
import pandas as pd

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        Input: Folder
        """)

    parser.add_argument(
        "-t", "--tdir",
        required=True, nargs="+",
        help="Folder with task files",
        )

    parser.add_argument(
        "-f", "--fdir",
        required=True, nargs="+",
        help="Folder with fmri files",
        )

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Folder where output file is saved",
        )

    parser.add_argument(
        "-v", "--verbose",
        required=False, nargs="+",
        help="Verbose to get more information about what's going on",
        )

    args =  parser.parse_args()
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def get_ids(taskDir, fmriDir):
    if not os.path.exists(taskDir):
        sys.exit('The task folder doesnt exist: {}'.format(iDir))
        return
    if not os.path.exists(fmriDir):
        sys.exit('The fMRI folder doesnt exist: {}'.format(iDir))
        return
    tFiles = glob.glob(os.path.join(taskDir,'sub*.tsv'))
    tIDs = []
    for tfile in tFiles:
        tfilename = os.path.basename(tfile)
        tid = tfilename.split('-')[1].split('_')[0]
        tIDs.append(tid)

    fFiles = glob.glob(os.path.join(fmriDir,'fmri*.nii'))
    fIDs = []
    for ffile in fFiles:
        ffilename = os.path.basename(ffile)
        fid = ffilename.split('_')[1].split('sub')[1]
        fIDs.append(fid)

    ids = list(set(tIDs) & set(fIDs))
    return ids

def main():
    args = get_arguments()
    output_dir = args.odir[0]
    ids = get_ids(args.tdir[0], args.fdir[0])
    data_ids = pd.DataFrame({'sub_ids' : ids})
    #print(data_ids['sub_ids'])
    data_ids.sort_values(by = ['sub_ids'], axis = 0, ascending = True, inplace= True)
    #print(data_ids.iloc[:, 0])
    j = 0
    for i in range (0, 4):
        data = data_ids.iloc[j:(j+24), :]
        data.to_csv(output_dir+'/sublist_'+str(i+1)+'.tsv', sep = '\t',
        header=True, index=False)
        j = j + 24

if __name__ == '__main__':
    sys.exit(main())
