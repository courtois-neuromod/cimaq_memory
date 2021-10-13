
import os
import sys
import argparse
import glob
import re
import numpy as np
import scipy
import pandas as pd
from numpy import nan as NaN
from scipy.stats import norm

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
        help="Output    folder - if doesnt exist it will be created",
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
#List of Behav output scans
def get_all_bFiles(iDir):
    if not os.path.exists(iDir):
        sys.exit('This folder doesnt exist: {}'.format(iDir))
        return
    all_bFiles = glob.glob(os.path.join(iDir,'PostScan*'))
    print(len(all_bFiles))
    return all_bFiles

def get_ids(bFile_path):
        bFile = os.path.basename(bFile_path)
        #bNum = re.findall('bID(.+?)_', bFile)[0]
        #sNum = re.findall('_mriID(.+?).tsv', bFile)[0]
        bNum = re.findall('pscid(.+?)_', bFile)[0]
        sNum = re.findall('_dccid(.+?).tsv', bFile)[0]
        ids = [bNum, sNum]
        return ids

def get_subject_score(s_file):
    ids = get_ids(s_file)
    print(ids)

    s_data = pd.read_csv(s_file, sep='\t')
    # get counts
    hits = len(s_data[s_data['recognition_performance']=='Hit'].index)
    #if no miss trials:
    try:
        miss = len(s_data[s_data['recognition_performance']=='Miss'].index)
    except:
        miss = 0
    try:
        falseA = len(s_data[s_data['recognition_performance']=='FA'].index)
    except:
        falseA = 0
    corRej = len(s_data[s_data['recognition_performance']=='CR'].index)
    # calculate recognition scores
    old = hits + miss
    new = corRej + falseA
    HminFA = (hits/old) - (falseA/new)
    Z_probHit = norm.ppf(hits/old)
    Z_probFA = norm.ppf(falseA/new)
    dprime = Z_probHit - Z_probFA

    # get position scores
    try:
        corSou = len(s_data[s_data['position_accuracy']==2.0].index)
    except:
        corSou = 0
    wrongSou = len(s_data[s_data['position_accuracy']==1.0].index)
    AssoMem = corSou/(wrongSou+falseA)

    #calculate mean reaction times to item recognition
    #rt recognition: 0 = CR, 1 = FA, 2 = Hit, 3 = Miss (alphabetical)
    ri1 = s_data.groupby('recognition_performance').mean().recognition_responsetime

    if falseA==0:
        tempCR = ri1[0] #CR value
        tempH = ri1[1] #Hit value
        if miss==0:
            tempM = NaN
        else:
            tempM = ri1[2]
        ri1 = [tempCR, NaN, tempH, tempM]
    elif miss==0:
        tempCR = ri1[0] #CR value
        tempFA = ri1[1] #FA value
        tempH = ri1[2] #Hit value
        ri1 = [tempCR, tempFA, tempH, NaN]

    #rt recognition: 0 = New, 1 = OLD (alphabetical)
    ri2 = s_data.groupby('old_new').mean().recognition_responsetime
    #rt recognition: 0 = 0.0 (miss), 1 = 1.0 (wrong source),
    #2 = 2.0 (correct source) (alphabetical)
    ri3 = s_data.groupby('position_accuracy').mean().recognition_responsetime

    if miss ==0:
        tempWS = ri3[1.0]
        if corSou == 0:
            tempCS = NaN
        else:
            tempCS = ri3[2.0]
        ri3 = [NaN, tempWS, tempCS]
    elif corSou ==0:
        tempMi = ri3[0.0]
        tempWS = ri3[1.0]
        ri3 = [tempMi, tempWS, NaN]

    #calculate mean reaction times to source recognition
    #rt source: 0 = CR, 1 = FA, 2 = Hit, 3 = Miss (alphabetical)
    #Note that only 1= false alarm and 2 = hits are valid
    rs1 = s_data.groupby('recognition_performance').mean().position_responsetime
    if falseA ==0:
        tempCRs = rs1[0]
        tempHs = rs1[1]
        rs1 = [tempCRs, NaN, tempHs]
    #rt source: 0 = 0.0 (miss), 1 = 1.0 (wrong source),
    #2 = 2.0 (correct source) (alphabetical)
    #Note that only 1 = wrong source and 2 = correct source are valid
    rs3 = s_data.groupby('position_accuracy').mean().position_responsetime

    if miss ==0:
        tempWSs = rs3[1.0]
        if corSou == 0:
            tempCSs = NaN
        else:
            tempCSs = rs3[2.0]
        rs3 = [NaN, tempWSs, tempCSs]
    elif corSou ==0:
        tempMis = rs3[0.0]
        tempWSs = rs3[1.0]
        rs3 = [tempMis, tempWSs, NaN]

    data_list = [ids[0], ids[1], hits, miss, falseA, corRej, old, new,
    HminFA, Z_probHit, Z_probFA, dprime, corSou, wrongSou, AssoMem,
    ri1[0], ri1[1], ri1[2], ri1[3], ri2[0], ri2[1], ri3[1], ri3[2],
    rs1[1], rs1[2], rs3[1], rs3[2]]

    #print(data_list)
    return data_list


def extract_results(file_list):

    bData = pd.DataFrame()    #empty pandas DataFrame
    colNames = ['pscID', 'dccID', 'hits', 'miss', 'false_alarms', 'correct_rej',
    'old', 'new', 'Hits_minus_FA', 'Z_probabHit', 'Z_probabFA', 'dprime',
    'correct_source', 'wrong_source', 'associative_memScore', 'rt_reco_CR',
    'rt_reco_FA', 'rt_reco_Hit', 'rt_reco_Miss', 'rt_reco_New', 'rt_reco_Old',
    'rt_reco_wrongSource', 'rt_reco_correctSource', 'rt_source_FA', 'rt_source_Hit',
    'rt_source_wrongSource', 'rt_source_correctSource', ]

    for i in range(0, len(colNames)):
        bData.insert(loc=len(bData.columns), column=colNames[i], value=NaN,
        allow_duplicates=True)

    for bFile in file_list:
        #print(bFile)
        scores = get_subject_score(bFile)
        bData = bData.append(pd.Series(scores, index=bData.columns ), ignore_index=True)

    return bData

def main():
    args = get_arguments()
    all_bFiles = get_all_bFiles(args.idir[0])
    output_dir = os.path.join(args.odir[0], 'TaskResults')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    taskResults = extract_results(all_bFiles)
    taskResults.to_csv(output_dir+'/fMRI_behavMemoScores.tsv',
    sep='\t', header=True, index=False)

if __name__ == '__main__':
    sys.exit(main())
