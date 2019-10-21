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
        "-d", "--idir",
        required=True, nargs="+",
        help="Folder with input files",
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

def get_ids(fileDir):
    if not os.path.exists(fileDir):
        sys.exit('This folder doesnt exist: {}'.format(iDir))
        return
    files = glob.glob(os.path.join(fileDir,'sub*.tsv'))
    ids = []
    for file in files:
        filename = os.path.basename(file)
        id = filename.split('-')[1].split('_')[0]
        ids.append(id)
    return ids

def main():
    args = get_arguments()
    output_dir = args.odir[0]
    ids = get_ids(args.idir[0])
    data_ids = pd.DataFrame({'sub_ids' : ids})
    print(data_ids['sub_ids'])
    data_ids.sort_values(by = ['sub_ids'], axis = 0, ascending = True, inplace= True)
    print(data_ids.iloc[:, 0])
    data_ids.to_csv(output_dir+'/sub_list.tsv', sep='\t',
    header=True, index=False)

if __name__ == '__main__':
    sys.exit(main())
