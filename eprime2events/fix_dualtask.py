import os
import pandas as pd
from argparse import ArgumentParrser
from typing import Union
from get_desc import get_desc
from load_recursive import load_recursive

exclude = lambda exc, l: [i for i in l if all(s not in i for s in exc)]
include = lambda inc, l: [i for i in l if any(s in i for s in inc)]

def fix_dualtask(bids_dir: Union[str,os.PathLike])-> None:
    """
    Repair Onset files where two visits are concatenated.

    Some Onset files were not fixed by fix_dupindex.py as their
    indexes were not duplicated, but rather concatenated along
    the rows (x) axis. This script repairs these files by finding
    which half matches the corresponding Output file.

    Args:
        bids_dir: str
            Path to the BIDS directory.

    Returns:
        None
    """

    bads=[itm for itm in include('Onset', load_recursive(bids_dir))
          if pd.read_csv(os.path.join(bids_dir, itm),sep='\t').shape[0]==240]
    subdirs=[os.path.join(os.path.dirname(os.path.dirname(bad)),
                          os.path.dirname(bad))
             for bad in bads]
    output_paths=[os.path.join(subdir,exclude(['Retrieval'],
                                              include(['Output'],
                                                      os.listdir(subdir)))[0])
                  for subdir in subdirs]
    to_fix=tuple(zip(bads,output_paths))
    for itm in to_fix:
        broken=pd.read_csv(itm[0],sep='\t').drop(['Unnamed: 0', '0'],
                                                 axis=1).fillna('NA')
        outputs=pd.read_csv(itm[1], sep='\t').drop(['Unnamed: 0'],
                                                   axis=1).fillna('NA')
        broken = broken.rename(dict(tuple((itm[1],itm[0])
                                           for itm in
                                          enumerate(broken.columns))),
                               axis=1)
        upper=broken.iloc[:int(broken.shape[0]/2),:]
        bottom = broken.iloc[int(broken.shape[0]/2):,:]
        try:
            good=[itm for itm in [upper,bottom]
                  if itm[2].unique().all()==outputs['OldNumber'].unique().all()][0]
            good.to_csv(itm[0],sep='\t', encoding='UTF-8-SIG')
            print(good)
        except IndexError:
            print(upper[2].unique(),
                  bottom[2].unique(),
                  outputs['OldNumber'].unique())

def main()
    desc, help_msgs=get_desc(fix_dualtask)
    parser = ArgumentParrser(prog=fix_dualtask, usage=fix_dualtask.__doc__),
                             desc=desc)
    parser.add_argument('bids_dir', nargs=1, help=help_msgs[0])
    args=parser.parse_args()
    fix_dualtask(args.bids_dir[0])

if __name__ == '__main__':
    main()
