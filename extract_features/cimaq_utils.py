#!/usr/bin/python3

########################################################################
import os
import pandas as pd
from functools import reduce
from typing import Iterable, Sequence, Union
from unidecode import unidecode

byvalue = lambda dc, v: list(dc.keys())[list(dc.values()).index(v)]
dpaths = lambda d: [i[0] for i in os.walk(d)]
empty_not = lambda d: list(filter(os.listdir, dpaths(d)))
empty_only = lambda d: exclude(empty_not(d), dpaths(d))
find_key = lambda d,v: next((k for k, v in d.items() if val == v), None)
get_uids = lambda lst, s=0: dict(tuple(enumerate(sorted(list(lst)), s)))
intersect = lambda l1, l2: [v for v in tuple(set(l1)) if v in set(l2)]
lst_exc = lambda exc, l1: [i for i in l1 if i not in exc]
lst_inc = lambda inc, l1: [i for i in l1 if i not in inc]
pipe = lambda funcs, val: reduce(lambda res, f: f(res), funcs, val)
# Source:https://stackoverflow.com/questions/39123375/apply-a-list-of-python-functions-in-order-elegantly
prntonly = lambda s: ''.join([c for c in list(s) if c.isprintable()])
revdict = lambda d: dict(tuple((i[1], i[0]) for i in tuple(d.items())))
seq_eve = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 == 0)
seq_odd = lambda it: tuple(i[1] for i in enumerate(it) if i[0] % 2 != 0)
str_exc = lambda exc, l: [i for i in l if all(s not in i for s in exc)]
str_inc = lambda inc, l: [i for i in l if any(s in i for s in inc)]
s2sq = lambda s: [[s] if isinstance(s, str) else list(s)][0]
upath = lambda src: unidecode(re.sub('\s{1,}', '_', src))

def absoluteFilePaths(src:Union[str,os.PathLike]) -> list:
    """
    Return all absolute file paths in ``src`` recursively.

    Source:https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
    """

    for dpath,_,fnames in os.walk(src):
        for f in fnames: yield os.path.abspath(os.path.join(dpath, f))

def flatten(nested_seq: Union[Iterable, Sequence]) -> list:
    return [bottomElem for sublist in nested_seq for bottomElem
            in (flatten(sublist)
                if (isinstance(sublist, Sequence) \
                    and not isinstance(sublist, str))
                else [sublist])]

def sizeof_fmt(num, suffix='B') -> str:
    """
    Return the size of the input in a humanly readable manner.

    Source: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def get_default_args(func: callable) -> dict:
    """
    Return a dict containing the default arguments of ``func``.

    Args:
        func: callable
            Callable function from which to retrive its default parameters.

    Returns: dict
        Dict containing a function's default parameters.

    Notes:
    https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    import inspect
    import sys
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty}
# def megamerge(dflist: list, howto: str = "outer", onto: str = None) -> object:
#     """ Returns a pd.DataFrame made from merging the frames in 'dflist' """
#     return reduce(lambda x, y: pd.merge(
#         x, y, on=onto, how=howto).astype("object"), dflist)
