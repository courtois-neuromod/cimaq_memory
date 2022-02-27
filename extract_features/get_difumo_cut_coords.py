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


def get_difumo_cut_coords(dimension: int,
                          resolution_mm: int,
                          data_dir: Union[str, PathLike,
                                          PosixPath] = None,
                          output_dir: Union[str, PathLike,
                                            PosixPath] = None,
                          as_dataframe: bool = True
                          ) -> None:
    """
    Return each atlas ROI's in a DiFuMo atlas (4th axis) MNI coordinates as triplets.
    
    The returned triplets can be passed to a ``NiftiSpheresMasker`` instance.
    
    Args:
        dimension: int
            Desired number of ROIs in the map.
            Valid choices are 64, 128, 256, 512 and 1024.
            Note that runtime increases proportionally with
            the number of dimensions.

        resolution_mm: int
            Desired voxel size in mm.
            Valid choices are 2 or 3.

        data_dir: str or os.PathLike
            Directory where the atlases are located.
            If ``None`` (default) is provided, the current
            working directory is used.            
            If the requested atlas is not present, it will be downloaded with
            ``nilearn.datasets.fetch_atlas_difumo``.

        output_dir: str or os.PathLike (Default = None)
            Directory where to save the coordinates DataFrame as a csv or tsv file.
            If ``None`` (default) is provided, the extracted coordinates
            are not saved to disk.
        
        as_dataframe: bool (Default=True)
            When True (default), returns the constructed DataFrame.
            Returns ``None`` otherwise.
    
    Returns: pd.DataFrame or None
        Return value depends on the ``as_dataframe`` parameter.
    """
    
    if data_dir is None:
        data_dir = os.getcwd()    

    # Load DiFuMo Atlas map as Nifti image & labels as pandas DataFrame
    difumo = get_difumo(data_dir=data_dir, dimension=dimension,
                        resolution_mm=resolution_mm)

    # Find Each Atlas ROI's MNI Coordinates
    difumo_cut_coords = niplot.find_probabilistic_atlas_cut_coords(nimage.load_img(difumo.maps))

    # Use the Coordinates as Seeds for the NiftiSpheresMasker
    spheres_masker = NiftiSpheresMasker(seeds=difumo_cut_coords,standardize=True)

    # Save the Coordinates and ROI Labels to csv
    # Coordinates should be equivalent respective to each atlas map images
    difumo_coords = pd.DataFrame(difumo_cut_coords, columns=['x','y','z'],
                                 index=difumo.labels.difumo_names).reset_index()
    difumo_coords.set_axis(range(1,difumo.labels.shape[0]+1), axis=0, inplace=True)
    difumo_coords.reset_index(drop=False, inplace=True)
    difumo_coords.set_axis(['component', 'difumo_names', 'x', 'y', 'z'],
                               axis=1, inplace=True)
    difumo_coords = difumo_coords.set_index('component')
    
    if output_dir is not None:
        fname_base = f'difumo_{dimension}_dims_{str(resolution_mm)}mm'
        savename = '_'.join([fname_base, 'cut_coords.tsv'])
        difumo_coords.to_csv(os.path.join(output_dir, savename),
                             sep='\t', index='component', encoding='UTF-8-SIG')
    if as_dataframe is True:
        return difumo_coords


def main():
    desc, help_msgs = get_desc(get_difumo_cut_coords.__doc__)
    _parser = ArgumentParser(prog=get_difumo_cut_coords,
                             description=desc.splitlines()[0],
                             usage=desc)
    _parser.add_argument('dimension', dest=dimension, type=int,
                         nargs=1, help=help_msgs[0])
    _parser.add_argument('resolution_mm', dest=resolution_mm, type=int,
                         nargs=1, help=help_msgs[1])
    _parser.add_argument(*('-d', '--data-dir'), dest=data_dir,
                         default=None, nargs='*', help=help_msgs[2])
    _parser.add_argument(*('-o', '--output-dir'), dest=output_dir,
                         default=None, nargs='*', help=help_msgs[3])
    _parser.add_argument('as-dataframe', dest=as_dataframe, type=bool,
                         action='store_true', nargs='*', help=help_msgs[-1])
    args = _parser.parse_arguments()
    get_difumo_cut_coords(args.dimension[0], args.resolution_mm[0], args.data_dir,
                          args.output_dir, args.as_dataframe)

if __name__ == '__main__':
    main()
