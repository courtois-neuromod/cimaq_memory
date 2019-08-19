import os
import re
import sys

import glob

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
        help="Path to id_list.tsv, a list of of subject ids",
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
    npsych_list = glob.glob(os.path.join(npsych,'.csv'))
    return npsych_list

def get_cols(tname):
        """ Takes the name of a neuropsych test and returns a tupple of two lists.
        The first list has the headers of the columns that correspond to metrics of
        interest in the test's  table of scores.
        The second list has the names to be given to these metrics
        in the final table of neuropsych test scores outputed by this script
        Parameter:
        ----------
        tname: string (name of a neuropsych test)

        Return:
        ----------
        a tupple associated with the test's name (key in the switcher dictionnary);
        the tupple is two lists of strings of equal length
        """
    switcher = {
        'alpha_span': (['71233_rappel_alpha_item_reussis', '71233_rappel_alpha_pourcentage'], ['aspan_rappel_item_reussis', 'aspan_rappel_pourcentage']),
        'boston_naming_test': (['57463_boston_score_correcte_spontanee', '57463_boston_score_total'],['boston_correcte_spontanee', 'boston_total']),
        'easy_object_decision': (['45463_score'], ['easy_object_score']),
        'echelle_depression_geriatrique': (['    d.70664_score'], ['gds_score']),
        'echelle_hachinski': (['86588_score'],['hachinski_score']),
        'evaluation_demence_clinique': (['34013_cdr_sb'], ['cdr_sb']),
        'fluence_verbale_animaux': (['    18057_score_reponse_correcte'], ['verb_flu_correct_responses']),
        'histoire_logique_de_wechsler_rappel_differe': (['40801_score_hist_rappel_differe'],['log_story_delayed_recall']),
        'memoria': (['18087_score_libre_correcte', '18087_score_indice_correcte'],['memoria_libre_correct', 'memoria_indice_correct']),
        'moca': (['12783_score', '12783_score_scolarite'], 'moca_score', 'moca_score_scolarite'),
        'prenom_visage': (['33288_score_rappel_immediat', '33288_score_rappel_differe'], ['prenom_visage_rappel_immediat', 'prenom_visage_rappel_differe']),
        'test_enveloppe': (['75344_score_memoire_prospective', '75344_score_memoire_retrospective'], ['env_memoire_prospective', 'env_memoire_retrospective']),
        'tmmse': (['80604_score_total'],['mmse_total']),
        'trail_making_test': (['44695_temps_trailA', '44695_temps_trailB'],['trailA_time', 'trailB_time'])

    }
    return switcher.get(tname, ([], []))

def extract_npsych(ids, npsych, output):
    neuro_scores = pd.Dataframe() # create blank dataframe
    iso_list = ['alpha_span', 'moca']
    npsych_tests = get_tests(npsych)
    for test in npsych_tests:
        if test in isolist:
            test_df = pd.read_csv(test, sep = ',', encoding='ISO-8859-1')
        else:
            test_df = pd.read_csv(test, sep = ',')
        # only keep scores from subjects who did the fMRI task_files
        # sort scores by subject id
        test_df = test_df[test_df['CandID'] in ids].copy()
        test_df.sort_values(by = ['CandID'], axis = 0, ascending = True, inplace= True)

        # only keep columns with metrics of interest
        tname = re.findall('[0-9]+_(.+?).csv', os.path.basename(test))[0]
        cols = get_cols(tname) # return list of column headers to include
        cols_current = cols[0]
        cols_new = cols[1]
        cols_keep = ['CandID']
        for i in range (0, len(cols_current)):
            test_df.rename(columns={cols_current[i]: cols_new[i]}, inplace=True)
            col_keep.append(cols_new[i])
        test_df = test_df[cols_keep]

        # Add values to main dataframe for corresponding subject...
    # (listed on ids list)
    #and order table by subject id.
    # 'PSCID': column 0; 'CandID': column 1
        # load test as pandas, keep subjects that match list,
        # order rows per subject, and keep columns w metrics of interest
        # insert columns into dataframe
    return neuro_scores  # or export and save it?


def main():
    args =  get_arguments()
    ids = pd.read_csv(args.sdir[0], sep = '\t')['sub_ids'] # sub_list.tsv, a list of subject ids in .tsv format
    npsych_dir = args.ndir[0]
    output_dir = os.path.join(args.odir[0], 'Neuropsych')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    extract_npsych(ids, npsych_dir, output_dir)

if __name__ == '__main__':
    sys.exit(main())
