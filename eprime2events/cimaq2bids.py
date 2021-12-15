#!/usr/bin/python3

from argparse import ArgumentParser

import os
import pandas as pd
import re
from io import StringIO
from os.path import basename as bname
from typing import Union
from unidecode import unidecode
from zipfile import ZipFile

from fix_dupindex import fix_dupindex
from get_desc import get_desc
from get_encoding import get_encoding
# old == 9, new == 0
# old == 1, new == 2

def findbyvalues(matchdict:dict,
                 table: pd.DataFrame) -> dict:
    from flatten import flatten
    table = table.copy(deep=True).fillna('NA').astype(str)
    patterns = [re.compile(pattern+'|NA', re.I)
                for pattern in tuple(matchdict.values())]
    names = tuple(matchdict.keys())
    matchlist = []
    [matchlist.append([col[1] for col
                      in enumerate(table.columns)
                      if table[col[1]].str.match(pat).all()])
     for pat in patterns]
    matchcols = sorted(set(flatten(matchlist)))
    renamer = dict(tuple(zip(matchcols, names)))
    return renamer

def read_json(src:Union[str, os.PathLike])->dict:
    import json
    with open(src, 'r') as jfile:
        jobj = json.load(jfile)
        jfile.close()
    return jobj

def get_recog_performance(retile:pd.DataFrame)->list:
    score = ['Hit' if bool(row[1].Spatial_ACC == 1.0 and \
             row[1].Recognition_ACC == 1.0) else
             'CR' if bool('New' in row[1].OldNumber and
                          row[1].Recognition_RESP == 2.0) else
             'FA' if bool('New' in row[1].OldNumber and
                          row[1]['Recognition_RESP'] == 1.0)
             else 'Miss' for row in retile.iterrows()]
    return score

def filter_cimaq(src:Union[str, os.PathLike])->tuple:
    from get_znames import get_znames
    archpaths = sorted([os.path.join(src, afile)
                        for afile in os.listdir(src)])
    pattern = re.compile('Onset|Output', re.I)
    archinfos = pd.DataFrame(((bname(apath),
                                [item for item in get_znames(apath)
                                 if re.findall(pattern, item) and
                                 re.findall('\d{7}', item)[0] in item],
                                len([item for item in get_znames(apath)
                                 if re.findall(pattern, item) and
                                 re.findall('\d{7}', item)[0] in item]) > 3,
                               len([item for item in get_znames(apath)
                                 if re.findall(pattern, item) and
                                 re.findall('\d{7}', item)[0] in item]) < 3)
                               for apath in archpaths),
                             columns=['archname', 'required_files',
                                      'has_more_files', 'has_less_files'])
    archinfos['valid'] = [len(rfiles) == 3 for rfiles
                          in archinfos.required_files]
    invalid = archinfos.where(~archinfos.valid).dropna(axis=0)
    invalid['reason'] = ['has_more_files' if row[1].has_more_files else
                         'has_less_files' for row in invalid.iterrows()]
    archinfos = archinfos.where(archinfos.valid).dropna(axis=0)
    return archinfos, invalid

def cimaq2bids(src:Union[str, os.PathLike],
               matchdict:Union[str, os.PathLike],
               enc_coldict:Union[str, os.PathLike]=None,
                 ret_coldict:Union[str, os.PathLike]=None,
                 dst:Union[str,os.PathLike]=None):
    """
    Repair CIMA-Q task files and convert to BIDS format directly from zip file.

    All while inventorying not working archives, creates the BIDS structure
    with appropriate naming schemes within the destination directory.
    The original files remain untouched.

    Args:
         src: str, os.PathLike
             Path to directory containing the zip files.
         matchdict: str, os.PathLike
             Path to a json file containg a dict with column names as keys
             and corresponding regex pattern as values.
         enc_coldict: str, os.PathLike, optional
             Path to a json file containg a dict with column names as keys
             and new column names as values. Used for the Encoding task.
             If provided, the final number of columns must
             match the lenght of the dict.
         ret_coldict: str, os.PathLike, optional
             Path to a json file containg a dict with column names as keys
             and new column names as values. Used for the Retrieval task.
             If provided, the final number of columns must
             match the lenght of the dict.
         dst: str, os.PathLike, optional
             Path to a destination direcctory where to create the
             BIDS structure. If None is provided, a directory of the same name
             as ``src`` is created in the system-default temporary directory.
    Returns: None

    Notes:
        Note 0: Clean up eprime programming mistake (St-Laurent, 2019)
            Replace position_response and position_responsetime values with NaN
            if subject perceived image as 'new' (position was not probed).
            There should be no response or RT value there.
            Values were carried over from previous trial (not reset in eprime).
            CONFIRMED w Isabel: subject must give a position answer when
            probed (image considered OLD) before eprime moves to the next trial.
        Note 1: incomplete or archive list
            An additional file, incomplete_archives.tsv, is created in the
            destination directory. It lists each archive and the specific
            reason why it was not working.
    """

    import numpy as np
    import tempfile
    from get_encoding import get_encoding

    archinfos, invalid = filter_cimaq(src)
    archinfos['required_files'] = [sorted(row[1]['required_files']) for
                                   row in archinfos.iterrows()]
    dst = [dst if dst is not None else
           os.path.join(tempfile.gettempdir(),
                        bname(src))][0]
    matchdict, wrong = read_json(matchdict), []
    for row in archinfos.iterrows():
        sub_id, ses_id = row[1].archname.split('_')[:2]
        sub_id, ses_id = 'sub-'+sub_id, 'ses-'+ses_id
        with ZipFile(os.path.join(src, row[1].archname)) as zf:
            taskfiles = [zf.read(name).decode(get_encoding(zf.read(name)))
                         for name in sorted(row[1].required_files)]
            zf.close()
        ons, out, ret = taskfiles
        ons = fix_dupindex(pd.read_fwf(StringIO(unidecode(ons)),
                                        header=None))
        out = pd.read_csv(StringIO(unidecode(out)), sep='\t')
        ret = pd.read_csv(StringIO(unidecode(ret)), sep='\t')
        if ons.shape[0] != 120: # Data integrity checkups
            row[1]['reason']=' '.join([bname(row[1].required_files[0]),
                                       'would have (even after repairs)',
                                       f'{ons.shape[0]} rows'])
            row[1]['valid'] = False
            wrong.append(row[1])
            archinfos.drop(row[0])
            continue
        if tuple(ret.Recognition_RESP.unique()) == ():
            row[1]['reason']=' '.join([f'{bname(row[1].required_files[-1])}',
                                       'has no Recognition_RESP column'])
            row[1]['valid'] = False
            wrong.append(row[1])
            archinfos.drop(row[0])
            continue
        if 'OldNumber' not in ret.columns:
            row[1]['reason']=' '.join([f'{bname(row[1].required_files[-1])}',
                                       'has no OldNumber column'])
            row[1]['valid'] = False
            wrong.append(row[1])
            archinfos.drop(row[0])
            continue
        ons = ons.rename(findbyvalues(matchdict, ons), axis=1)
        ons = ons[['Onset', 'Offset', 'Isi']]
        ons['Duration'] = ons['Offset']-ons['Onset']
        out = out[['TrialNumber','Category', 'OldNumber',
                     'CorrectSource', 'Stim_RT']]
        out['Category'] = out['Category'].str.title()
        out['Stim_RT'] = out['Stim_RT'].div(1000)
        out = pd.concat([out, ons], axis=1)
        out = out.iloc[3:, :]
#        if out.shape[0] != ret.shape[0]:
#        out, ret = out.reset_index(drop=True), ret.reset_index(drop=True)
#        out = out.iloc[:,:]
#        if out.shape[0] != ret.shape[0]:
#            out = out.iloc[:ret.index[-1],:]
#        longest = [df for df in [out,ret] if df.shape[0] ==
#                    max((out.shape[0],ret.shape[0]))][0]
#            shortest = [df.index for df in [out,ret] if df.shape[0] ==
#                        min((out.shape[0],ret.shape[0]))][0]
#            out, ret = out.iloc[shortest,:], ret.iloc[shortest,:]
#        longest = longest.iloc[shortest.index, :]
#        shortest = shortest.iloc[shortest.index, :]
#        if out.shape[0] != ret.shape[0]:
#            out = out.iloc[ret.index]
#        out = out.iloc[int(out.shape[0]-ret.shape[0]):, :]
        ret = ret[['Stim', 'OldNumber', 'Recognition_RESP',
                     'Recognition_RT', 'Spatial_RESP', 'Spatial_RT']]

        # Fix Recognition_RESP alternative values
        if tuple(ret.Recognition_RESP.unique()) in ((0,9), (9,0)):
            ret['Recognition_RESP'] = ret['Recognition_RESP'].replace({9:1,0:2})

        # Note 0: Eprime Mistake
        ret[['Recognition_RT','Spatial_RT']] = [(row[1]['Recognition_RT']/1000,
                                                 row[1]['Spatial_RT']/1000) if
                                                 row[1]['Recognition_RESP'] == 1
                                                 else (np.nan, np.nan) for
                                                 row in ret.iterrows()]
        ret['CorrectSource'] = ret.OldNumber.map(dict(zip(out.OldNumber,
                                                            out.CorrectSource)))
        ret['Recognition_ACC'] = [float(bool(row[1].Recognition_RESP == 2
                                        and 'New' in row[1].OldNumber
                                        or row[1].Recognition_RESP == 1
                                        and 'Old' in row[1].OldNumber))
                                   for row in ret.iterrows()]
        ret[['Spatial_RESP','Spatial_RT']] = \
            ret[['Spatial_RESP', 'Spatial_RT']].where(ret.Recognition_RESP!=2)

       # Experimental setup (``Spatial_RESP`` keys) were changed since V03
        alt_resp_keys = {1.0: 8.0, 2.0: 9.0, 3.0: 5.0, 4.0: 6.0}
        if 4 in ret.Spatial_RESP:
            ret.Spatial_RESP = ret.Spatial_RESP.replace(alt_resp_keys)
        ret['Spatial_ACC'] = [float(bool(row[1]['Spatial_RESP'] ==
                                        row[1]['CorrectSource']))
                               if row[1].Recognition_RESP == 1 else np.nan
                               for row in ret.iterrows()]
        ret.insert(loc=1, column='StimCategory',
                    value=[re.findall('[a-z]*[A-Z]*',
                                      stim.replace('hard_', ''))[0]
                           for stim in ret.Stim])
        ret['RecognitionPerformance'] = get_recog_performance(ret)
        for col in tuple(col for col in ret.columns
                         if col not in ('category', 'CorrectSource',
                                        'OldNumber')):
            out[col] = out.OldNumber.map(dict(zip(ret.OldNumber,
                                                    ret[col])))
        out, ret = out.round(1), ret.round(1)

        if enc_coldict is not None:
            out = out.rename(read_json(enc_coldict), axis=1)
        if ret_coldict is not None:
            ret = ret.rename(read_json(ret_coldict), axis=1)
        out['position_correct'] = out['position_correct'].astype(float)
        ret.insert(loc=0, column='old_new',
                    value=[re.sub('\d*', '', item)
                           for item in ret.stim_id])
        os.makedirs(os.path.join(dst, sub_id, ses_id), exist_ok=True)
        prefix = f'{sub_id}_{ses_id}_task-memory'
        out.to_csv(os.path.join(dst, sub_id, ses_id,
                                 prefix+'_events.tsv'),
                    sep='\t', index=None, encoding='UTF-8-SIG')
        ret.to_csv(os.path.join(dst, sub_id, ses_id,
                                 prefix+'_behavioural.tsv'),
                    sep='\t', index=None, encoding='UTF-8-SIG')
    invalid = pd.concat([invalid, pd.DataFrame(wrong)])
    invalid.to_csv(os.path.join(dst, 'invalid_archives.tsv'),
                   sep='\t', index=None, encoding='UTF-8-SIG')

def main():
    desc, helps = get_desc(cimaq2bids)
    parser = ArgumentParser(prog=cimaq2bids, usage=cimaq2bids.__doc__,
                          description=desc)
    parser.add_argument('src', nargs=1, help=helps[0])
    parser.add_argument('matchdict', nargs=1, help=helps[1])
    parser.add_argument('--enc-cols', dest='enc_coldict', nargs='?',
                        help=helps[2])
    parser.add_argument('--ret-cols', dest='ret_coldict', nargs='?',
                        help=helps[3])
    parser.add_argument('-d', '--dst', dest='dst', nargs='?', help=helps[4])
    args = parser.parse_args()
    cimaq2bids(args.src[0], args.matchdict[0], args.enc_coldict, args.ret_coldict,
               args.dst)
if __name__ == '__main__':
    main()
