#!/usr/bin/python3

import lzma
import os
import shutil
import tarfile
import tempfile
from argparse import ArgumentDefaultsHelpFormatter
from io import BytesIO
from pathlib import Path
from unidecode import unidecode
from typing import Union

dpaths = lambda d: [i[0] for i in os.walk(d)]

def to_tar_xz(src:Union[str,os.PathLike],
              dst:Union[str,os.PathLike]=None):
    """ Compress files or directories to '.tar.xz' archive.
    
    Args:
        src: str, os.PathLike
            Path of the file or directory to compress
        dst: str, os.PathLike (optional)
            Destination path
    
    Returns: None
    """

    cleanpath = lambda p: unidecode(Path(p).resolve().as_posix())
    dst = [dst if dst is not None else
           cleanpath(os.path.join(tempfile.gettempdir(),
                                   os.path.basename(src)))][0]
    lzma_filters = [{**{"id": lzma.FILTER_LZMA2, "preset": 7},
                    **{**{"preset": lzma.PRESET_EXTREME,
                          "mode": lzma.MODE_FAST}}}]
    tdir = tempfile.TemporaryDirectory(prefix=tempfile.gettempdir()+'/')
    if os.path.isdir(src):
        shutil.copytree(src=src, dst=os.path.join(tdir.name,
                                                  os.path.basename(dst)))
    else:
        shutil.copy(src=src, dst=os.path.join(tdir.name,
                                              os.path.basename(dst)))
    with tarfile.TarFile(name=os.path.join(tdir.name,
                                           os.path.basename(dst)+'.tar'),
                         mode='w', encoding='UTF-8-SIG') as onetar:
        [onetar.add(name, arcname=os.path.basename(name))
         for name in dpaths(tdir.name)
         if name != tdir.name]
        check00 = Path(onetar.name)
        check01 = check00.stat()
    onetar.close() 
    with lzma.LZMAFile(filename=dst+'.tar.xz', mode='w',
                       filters=lzma_filters) as lzfile:
        lzfile.write(check00.read_bytes())
        check = (Path(dst+'.tar.xz'), Path(dst+'.tar.xz').stat())
    lzfile.close()
    del tdir.name
    return check[0].read_bytes()

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(prog=to_tar_xz,
                            usage=to_tar_xz.__doc__,
                            description=to_tar_xz.__doc__.splitlines()[0],
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('src', nargs=1)
    parser.add_argument('-d', '--dst', dest='dst', nargs='?',
                        required=False, default=None)
    args = parser.parse_args()
    to_tar_xz(args.src[0], args.dst)

if __name__ == '__main__':
    main()
