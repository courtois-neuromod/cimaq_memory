#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import typing
from argparse import ArgumentParser
from pandas import DataFrame as df
from typing import NewType, Union
from get_desc import get_desc
from get_has_dupindex import get_has_dupindex
from read_data import read_data

eveseq = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 == 0)
oddseq = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 != 0)

def fix_dupindex(inpt: Union[bytes, bytearray, str, os.PathLike,
                             pd.DataFrame, typing.io, object],
                 colnames: Union[list, tuple, set] = None,
                 prnt: bool = False
                ) -> pd.DataFrame:
    """
    Fix data with duplicate values along the x (row) axis.
    
    First, splits data into even and odd rows and creates
    two sepatate DataFrames from these. 
    Second, checks if any columns is exactly the same in both tables.
    Returns the concatenation along the y axis of one of the whole
    DataFrames to the columns which were different between
    the even-rows-only and the odd-rows-only DataFrames.
    
    Args:
        inpt: bytes, bytearray, str, os.PathLike, pd.DataFrame,
              typing.io or object
            The data object to repair. If it is a DataFrame, it is
            used as is. Otherwise, <read_table> is called on inpt
            to create a DataFrame.
        colnames: list, tuple or set of [str or int], optional
            Names to give each column of the repaired DataFrame.
            Ideally, the number of names in colnames should match the
            number of columns of the result. However, no error is raised
            if it does not. Exceeding names will be discarded, and
            columns left unnamed are named by their index.
    
    Returns: pd.DataFrame
        The data without duplicate rows or columns.
    """
    if isinstance(inpt, pd.DataFrame):
        inpt = pd.DataFrame(inpt.values)
    else:
        inpt = pd.DataFrame(read_table(inpt).values)
    everows = df(eveseq(list(inpt.values))).fillna("NA").reset_index(drop=True)
    oddrows = df(oddseq(list(inpt.values))).fillna("NA").reset_index(drop=True)
    evecols, oddcols = everows.T.values.tolist(), oddrows.T.values.tolist()
    test = [oddcol[0] for oddcol in enumerate(oddcols)
            if not oddcol[1] in evecols]
    rprd = pd.concat([everows, oddrows.iloc[:, test]],
                     axis=1).replace('NA', np.nan)
    rprd = rprd.set_axis(list(range(rprd.shape[1])), axis=1, inplace=False)
    if prnt is True:
        print(rprd)
    return rprd

def main():
    InptType = NewType('InptType', [bytes, bytearray, str, os.PathLike,
                                    pd.DataFrame, object])
    ColType = NewType('ColType', [list, tuple, set])
    desc, help_msgs = get_desc(function_name=fix_dupindex)
    parser = ArgumentParser(prog=fix_dupindex, usage=fix_dupindex.__doc__)
    parser.add_argument('inpt', dest='inpt', type=InptType, help=help_msgs[0])
    parser.add_argument('-c', '--colnames', dest='colnames', type=ColType,
                        default=None, nargs='?', required=False, help=help_msgs[1])
    parser.add_argument('-p', '--print', dest='prnt', action='store_true',
                        help=help_msgs[-1])
    args = parser.parse_args()
    fix_dupindex(args.inpt[0], args.colnames, args.prnt)

if __name__ == '__main__':
    main()
                    
    
