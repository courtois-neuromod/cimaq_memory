import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from numpy import nan as NaN

def get_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        Input: Folder
        """)

    parser.add_argument(
        "-i", "--ifile",
        required=True, nargs="+",
        help="Folder with task files",
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

def get_all(subs, oDir):
    col = ['participant_id']
    alls = subs[col]
    numsub = alls.shape[0]
    alls.to_csv(os.path.join(oDir, 'list_all_'+str(numsub)+'subs.tsv'),
    sep = '\t', header=True, index = False)

def get_hit_miss(subs, num, oDir):
    hm = subs[subs['hits']>num]
    hm = hm[hm['miss']>num]
    col = ['participant_id']
    hm = hm[col]
    numsub = hm.shape[0]
    hm.to_csv(os.path.join(oDir, 'list_hit_miss_'+str(numsub)+'subs.tsv'),
                sep = '\t', header=True, index = False)

def get_cs_ws(subs, num, oDir):
    cw = subs[subs['correct_source']>num]
    cw = cw[cw['wrong_source']>num]
    col = ['participant_id']
    cw = cw[col]
    numsub = cw.shape[0]
    cw.to_csv(os.path.join(oDir, 'list_cs_ws_'+str(numsub)+'subs.tsv'),
                sep = '\t', header=True, index = False)

def get_cs_miss(subs, num, oDir):
    cs_miss = subs[subs['miss']>num]
    cs_miss = cs_miss[cs_miss['correct_source']>num]
    col = ['participant_id']
    cs_miss = cs_miss[col]
    numsub = cs_miss.shape[0]
    cs_miss.to_csv(os.path.join(oDir, 'list_cs_miss_'+str(numsub)+'subs.tsv'),
                sep = '\t', header=True, index = False)

def main():
    args = get_arguments()
    output_dir = args.odir[0]
    subs = pd.read_csv(args.ifile[0], sep = '\t')
    # exclude subjects who failed QC
    subs = subs[subs['QC_status']!= 'F']
    get_all(subs, output_dir)

    get_hit_miss(subs, 14, output_dir)
    get_cs_ws(subs, 14, output_dir)
    get_cs_miss(subs, 14, output_dir)

if __name__ == '__main__':
    sys.exit(main())
