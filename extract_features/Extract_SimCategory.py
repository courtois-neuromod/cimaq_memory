import os
import sys

import argparse
import glob
import logging
from numpy import nan as NaN
import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        Input: Folder with zip files
        """)

    parser.add_argument(
        "-d", "--idir",
        required=True, nargs="+",
        help="Folder to be sorted")

    parser.add_argument(
        "-o", "--odir",
        required=True, nargs="+",
        help="Output folder - if doesn\'t exist it will be created.")

    parser.add_argument(
        '--log_level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level of the logging class.')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args

def main():
    args = get_arguments()
    logging.basicConfig(level=args.log_level)
    oFolder = args.odir[0]
    iFolder = args.idir[0]

    all_events = glob.glob(os.path.join(iFolder, 'sub*_events.tsv'))
    print(len(all_events))

    # loop over event files
    for event in all_events:
        base = os.path.basename(event)
        id = base.split('-')[1].split('_')[0]
        e_file = pd.read_csv(event, sep = '\t')

        # keep only trials for which fMRI data was collected
        e_file = e_file[e_file['unscanned']==0]

        cat_col = ['stim_category']
        name_col = ['stim_name']

        e_cat = e_file[cat_col]
        e_stim = e_file[name_col]

        e_cat.to_csv(os.path.join(oFolder, 'sub-'+id+'_categories.tsv'),
                    sep = '\t', header=True, index = False)
        e_stim.to_csv(os.path.join(oFolder, 'sub-'+id+'_stimuli.tsv'),
                    sep = '\t', header=True, index = False)

if __name__ == '__main__':
    sys.exit(main())
