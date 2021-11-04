#!/usr/bin/python3

import os
import sys
from typing import Union

from load_recursive import load_recursive

exclude = lambda exc, l: [i for i in l if all(s not in i for s in exc)]
include = lambda inc, l: [i for i in l if any(s in i for s in inc)]

def load_cimaq_taskfiles(src: Union[str, os.PathLike]) -> tuple:
    """
    Load CIMA-Q event files (alphabetical order for each session.
    
    The original CIMA-Q event files being damaged, careful session
    by session and trial by trial matching is required. It is
    therefore advantageous to map each file path appropriately.
    This allows cleaning all files in a more homegeneous manner.

    Args:
        src: str, os.PathLike
            The top-level directory of the files, sorted
            according to BIDS Standard.

    Returns: tuple(tuple(str, str, str))
        A tuple of 3-items tuples (3 files per visit for each visit)
    """

    onsets = sorted(include(['Onset'],
                            [itm for itm in load_recursive(src)
                             if itm.endswith('.tsv')]))

    retfiles = sorted(include(['Retrieval'],
                              [itm for itm in load_recursive(src)
                               if itm.endswith('.tsv')]))

    outputs = sorted(exclude(retfiles+onsets,
                             [itm for itm in load_recursive(src)
                              if itm.endswith('.tsv')]))

    taskfiles = tuple(tuple(sorted(itm)) for itm
                      in list(zip(onsets, outputs, retfiles)))
    return taskfiles

def main():
    load_cimaq_taskfiles(sys.argv[1])

if __name__ == '__main__':
    main()

