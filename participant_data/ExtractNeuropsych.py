import os
import re
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
        Import columns of ordered neuropsych test scores for
        participants who performed the fMRI memory task
        """)

    parser.add_argument(
        "-s", "--sdir",
        required=True, nargs="+",
        help="Path to id_list.tsv, a list of subject ids",
        )

    parser.add_argument(
        "-n", "--ndir",
        required=True, nargs="+",
        help="Folder with tables of neuropsych scores (.csv)",
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

def get_tests(npsych):
    """ Returns a list of .csv tables with neuropsych scores
    (one row per participant)
    Parameter:
    ----------
    npsych: strings (path to neuropsych data folder)

    Return:
    ----------
    npsych_list (list of .csv files)
    """
    if not os.path.exists(npsych):
        sys.exit('This folder does not exist: {}'.format(tDir))
        return
    npsych_list = glob.glob(os.path.join(npsych,'*.csv'))
    return npsych_list

def get_cols(tname):
    """ Takes the name of a neuropsych test and returns a tupple of two lists.
    The first list has the headers of the metrics of
    interest (columns in the test's  score table).
    The second list has the names to be given to these metrics
    in the final table of neuropsych scores outputed by this script.
    Parameter:
    ----------
    tname: string (name of a neuropsych test)

    Return:
    ----------
    a tupple associated with the test's name (key in the switcher dictionnary);
    the tupple contains two lists of strings of equal length
    """
    switcher = {
        'alpha_span': (['71233_rappel_alpha_item_reussis', '71233_rappel_alpha_pourcentage'], ['aspan_recall_correct_items', 'aspan_recall_percentage']),
        'boston_naming_test': (['57463_boston_score_correcte_spontanee', '57463_boston_score_total'],['boston_correct_spontaneous', 'boston_total']),
        'easy_object_decision': (['45463_score'], ['easy_object_decision_score']),
        'echelle_depression_geriatrique': (['    d.70664_score'], ['gds_score']),
        'echelle_hachinski': (['86588_score'],['hachinski_score']),
        'evaluation_demence_clinique': (['34013_cdr_sb'], ['cdr_sb']),
        'fluence_verbale_animaux': (['    18057_score_reponse_correcte'], ['verb_flu_correct_responses']),
        'histoire_logique_wechsler_rappel_immediat': (['24918_score_hist_rappel_immediat'],['log_story_immediate_recall']),
        'histoire_logique_wechsler_rappel_differe': (['40801_score_hist_rappel_differe'],['log_story_delayed_recall']),
        'memoria': (['18087_score_libre_correcte', '18087_score_indice_correcte'],['memoria_free_correct', 'memoria_indice_correct']),
        'moca': (['12783_score', '12783_score_scolarite'], ['moca_score', 'moca_score_schooling']),
        'prenom_visage': (['33288_score_rappel_immediat', '33288_score_rappel_differe'], ['name_face_immediate_recall', 'name_face_delayed_recall']),
        'ravlt': (['86932_mots_justes_essai_1', '86932_mots_justes_essai_total1', '86932_mots_justes_rappel_diff_a', '86932_score_total_reconnaissance'], ['RAVLT_trial1', 'RAVLT_total', 'RAVLT_delRecall', 'RAVLT_recognition']),
        'test_enveloppe': (['75344_score_memoire_prospective', '75344_score_memoire_retrospective'], ['env_prospective_memory', 'env_retrospective_memory']),
        'tmmse': (['80604_score_total'],['mmse_total']),
        'trail_making_test': (['44695_temps_trailA', '44695_temps_trailB', '44695_ratio_trailB_trailA'],['trailA_time', 'trailB_time', 'trailB_trailA_ratio']),
        'stroop': (['77180_cond3_temps_total', '77180_cond3_total_erreurs_corrigees', '77180_cond3_total_erreurs_non_corrigees', '77180_cond4_temps_total', '77180_cond4_total_erreurs_corrigees', '77180_cond4_total_erreurs_non_corrigees'],['Stroop_cond3_time', 'Stroop_cond3_corr_errors', 'Stroop_cond3_nonCorr_errors', 'Stroop_cond4_time', 'Stroop_cond4_corr_errors', 'Stroop_cond4_nonCorr_errors']),
        'vocabulaire': (['87625_score'],['WAIS_vocabulary']),
        'wais_digit_symbol':(['12321_resultat_brut'],['WAIS_digit_symbol_total'])
    }
    return switcher.get(tname, ([], []))

def extract_npsych(ids, npsych, output):
    output_dir = os.path.join(output, 'subtests')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # create dataframe with column of subject ids (dccid)
    neuro_scores = pd.DataFrame()
    neuro_scores.insert(loc = 0, column = 'dccid', value = ids, allow_duplicates=False)
    neuro_scores.set_index('dccid', inplace = True)

    iso_list = ['alpha_span', 'moca', 'ravlt']
    skip_list = ['diagnostic_clinique', 'borb_line_orientation']
    npsych_tests = get_tests(npsych) # returns list of score tables
    for test in npsych_tests:
        # extract test name
        tname = re.findall('[0-9]+_(.+?).csv', os.path.basename(test))[0]
        print(tname)

        if tname not in skip_list:
            if tname in iso_list:
                test_df = pd.read_csv(test, sep = ',', encoding='ISO-8859-1')
            else:
                test_df = pd.read_csv(test, sep = ',')
            # make score dataframe indexed (searchable) by subject id
            test_df.set_index('CandID', inplace = True)

            # get columns of interests
            cols = get_cols(tname) # returns tupple (two lists)
            cols_current = cols[0] # headers of columns of interest
            cols_new = cols[1] # headers to give columns in final output

            # only keep columns of interest
            test_df = test_df[cols_current]
            test_df.to_csv(output_dir+'/'+tname+'.tsv', sep = '\t', header=True, index = True)

            # identify ids of subjects with entries in score table
            k = neuro_scores.index.intersection(test_df.index)
            ncol = len(cols_current)
            for i in range (0, ncol):
                neuro_scores.insert(loc = len(neuro_scores.columns), column = cols_new[i], value = NaN, allow_duplicates=True)
            # import data into final dataframe
                for j in k:
                    neuro_scores.loc[j, cols_new[i]] = test_df.loc[j, cols_current[i]]

    # add back subject id column as regular column
    neuro_scores.reset_index(level=None, drop=False, inplace=True)

    neuro_scores.insert(loc=neuro_scores.shape[1], column = 'memoria_total_correct', value = NaN,
    allow_duplicates=True)
    neuro_scores['memoria_total_correct'] = neuro_scores['memoria_free_correct'] + neuro_scores['memoria_indice_correct']
    neuro_scores.drop(['memoria_indice_correct'], axis = 1, inplace=True)

    # put columns in order, drop memoria_indice_correct column
    col_order = ['dccid', 'hachinski_score', 'cdr_sb',
    'mmse_total', 'moca_score', 'moca_score_schooling', 'gds_score',
    'WAIS_digit_symbol_total', 'trailA_time', 'trailB_time', 'trailB_trailA_ratio',
    'Stroop_cond3_time', 'Stroop_cond3_corr_errors', 'Stroop_cond3_nonCorr_errors',
    'Stroop_cond4_time', 'Stroop_cond4_corr_errors', 'Stroop_cond4_nonCorr_errors',
    'easy_object_decision_score', 'boston_correct_spontaneous', 'boston_total',
    'WAIS_vocabulary', 'verb_flu_correct_responses', 'aspan_recall_correct_items',
    'aspan_recall_percentage', 'env_prospective_memory', 'env_retrospective_memory',
    'memoria_free_correct', 'memoria_total_correct', 'name_face_immediate_recall',
    'name_face_delayed_recall', 'log_story_immediate_recall', 'log_story_delayed_recall',
    'RAVLT_trial1', 'RAVLT_total', 'RAVLT_delRecall', 'RAVLT_recognition']

    neuro_scores = neuro_scores[col_order]

    return neuro_scores

def main():
    args =  get_arguments()
    ids = pd.read_csv(args.sdir[0], sep = '\t')['sub_ids'] # sub_list.tsv, a list of subject ids in .tsv format
    npsych_dir = args.ndir[0]
    output_dir = os.path.join(args.odir[0], 'Neuropsych')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    npsych_table = extract_npsych(ids, npsych_dir, output_dir)
    npsych_table.to_csv(output_dir+'/ALL_Neuropsych_scores.tsv', sep = '\t', header=True, index = False)

if __name__ == '__main__':
    sys.exit(main())
