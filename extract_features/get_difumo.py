#!/usr/bin/python3

import os
import pandas as pd

from argparse import ArgumentParser
from io import StringIO
from os import PathLike
from pathlib import Path, PosixPath
from nilearn.datasets import fetch_atlas_difumo
from nilearn.image import load_img
from sklearn.utils import Bunch
from typing import Union

from get_desc import get_desc


def get_difumo(dimension: int,
               resolution_mm: int,
               data_dir: Union[str, PathLike,
                               PosixPath] = None
              ) -> Bunch:

    """
    Return a dict-like structure containing the DiFuMo atlas ROI maps and labels.
    
    Based on the ``nilearn.datasets.fetch_atlas_difumo`` function, it ensures
    that the labels are returned as a pandas DataFrame of strings (not bytes).

    Args:
        dimension: int
            Desired number of ROIs in the map.
            Valid choices are 64, 128, 256, 512 and 1024.
            Note that runtime increases proportionally with
            the number of dimensions.

        resolution_mm: int
            Desired voxel size in mm
            Valid choices are 2 or 3

        data_dir: str, os.PathLike or pathlib.PosixPath (Default=None)
            Directory where the atlases are located.
            If ``None`` (default) is provided, the current
            working directory is used.
            If the requested atlas is not present, it will be downloaded with
            ``nilearn.datasets.fetch_atlas_difumo``.

    Returns: Bunch
        Dict-like mapping with 'maps' and 'labels' as keys.
        The keys's respective values types are
        ``nibabel.nifti1.Nifti1Image`` (4D) and
        ``pandas.DataFrame`` (index=range(1, ``dimensions``, name='components').
    """

    suffix = f'*{dimension}/{resolution_mm}mm/*.nii.gz'
    if data_dir is None:
        data_dir = os.getcwd()
    maps_path = sorted(Path(data_dir).rglob(suffix))
    if maps_path == []:
        fetch_atlas_difumo(dimension, resolution_mm,
                           data_dir, resume=True)

    maps = list(Path(data_dir).rglob(f'*{dimension}/{resolution_mm}mm/*.nii.gz'))[0]
    labels = list(Path(data_dir).rglob(f'*{dimension}/*.csv'))[0]
    maps, labels = tuple(map(str, (maps, labels)))
    labels_buff = StringIO(Path(labels).read_bytes().decode('UTF-8').lower())
        
    maps = load_img(maps)
    labels = pd.read_csv(labels_buff)
    labels['component'] = labels['component'] - 1
    labels = labels.set_index('component')
    return Bunch(maps=maps, labels=labels)

def main():
    desc, help_msgs = get_desc(get_difumo.__doc__)
    _parser = ArgumentParser(prog=get_difumo,
                             description=desc.splitlines()[0],
                             usage=desc)
    _parser.add_argument('dimension', dest=dimension, type=int,
                         nargs=1, help=help_msgs[0])
    _parser.add_argument('resolution_mm', dest=resolution_mm, type=int,
                         nargs=1, help=help_msgs[1])
    _parser.add_argument(*('-d', '--data-dir'), dest=data_dir,
                         nargs='*', help=help_msgs[-1])
    args = _parser.parse_arguments()
    get_difumo(args.dimension[0], args.resolution_mm[0], args.data_dir)

if __name__ == '__main__':
    main()
