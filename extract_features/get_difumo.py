#!/usr/bin/python3

import os
import pandas as pd
from pathlib import Path
from sklearn.utils import Bunch
from typing import Union

def get_difumo(atlases_dir:Union[str,os.PathLike],
               n_dims:int, resolution_mm:int) -> Bunch:
    ndims, resolution_mm = tuple(map(str, (n_dims, resolution_mm)))
    maps = list(Path(atlases_dir).rglob(f'*{n_dims}/{resolution_mm}mm/*.nii.gz'))[0]
    labels = list(Path(atlases_dir).rglob(f'*{n_dims}/*.csv'))[0]
    maps, labels = tuple(map(str, (maps, labels)))
    maps = nilearn.image.load_img(maps)
    labels = pd.read_csv(labels).set_index('Component')
    return Bunch(maps=maps, labels=labels)
