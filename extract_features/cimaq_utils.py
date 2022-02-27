#!/usr/bin/python3

########################################################################
import os
import pandas as pd
import re
from functools import reduce
from typing import Iterable, Sequence, Union
from unidecode import unidecode


def byvalue(dc: dict, val) -> list:
    """
    Return key from ``dc`` if its value is equal to ``val``.
    """

    return list(dc.keys())[list(dc.values()).index(val)]


def chain_pipe(funcs: Sequence[callable], val: object) -> object:
    """
    Apply functions in ``funcs`` one by one to ``val``.

    Source:
    https://stackoverflow.com/questions/39123375/apply-a-list-of-python-functions-in-order-elegantly
    """

    return reduce(lambda res, f: f(res), funcs, val)


def dpaths(src: Union[str, os.PathLike]) -> list:
    """
    Return a sorted recursive list of absolute folders' paths in ``src``.
    """

    return sorted(i[0] for i in os.walk(src))


def empty_not(src: Union[str, os.PathLike]) -> list:
    """
    Return sorted recursive non-empty folders' absolute paths list in ``src``.
    """

    return sorted(filter(os.listdir, sorted(i[0] for i in os.walk(src))))


def empty_only(src: Union[str, os.PathLike]) -> list:
    """
    Return a sorted list of empty directories absolute paths in ``src``.
    """

    return lst_exc(empty_not(src), sorted(i[0] for i in os.walk(src)))


def find_key(dc: dict, val) -> object:
    return next((k for k, v in dc.items() if val == v), None)


def get_uids(lst: list, s: int = 0) -> dict:
    return dict(tuple(enumerate(sorted(list(lst)), s)))


def intersect(l1: Sequence, l2: Sequence) -> Sequence:
    """
    Return the intersection (unique common items) from ``l1`` & ``l2``.
    Source:
    https://stackoverflow.com/questions/3697432/how-to-find-list-intersection
    """

    return list(set(l1) & set(l2))


def lst_exc(exc: Sequence, seq: Sequence) -> Sequence:
    return [i for i in seq if i not in exc]


def lst_inc(inc: Sequence, l1: Sequence) -> Sequence:
    """
    Return the intersection (non-unique common items) from ``l1`` & ``l2``.

    Similar to ``set.intersection``, but duplicate items are not discarded.
    """

    return [i for i in l1 if i in inc]


def prntonly(txt: str) -> str:
    """
    Return str containing only UTF-8 printable characters in ``txt``.
    """

    return ''.join([c for c in list(txt) if c.isprintable()])


def revdict(dc: dict) -> dict:
    """
    Return dict inversely mapping key-value pairs in ``dc``.
    """

    return dict(tuple((i[1], i[0]) for i in tuple(dc.items())))


def seq_eve(it: object) -> tuple:
    return tuple(i[1] for i in enumerate(it) if i[0] % 2 == 0)


def seq_odd(it: object) -> tuple:
    return tuple(i[1] for i in enumerate(it) if i[0] % 2 != 0)


def str_exc(exc: Sequence, lst: Sequence) -> list:
    return [i for i in lst if all(s not in i for s in exc)]


def str_inc(inc: Sequence, lst: Sequence) -> list:
    return [i for i in lst if any(s in i for s in inc)]


def s2sq(txt: str) -> list:
    return [[txt] if isinstance(txt, str) else list(txt)][0]


def upath(src: Union[str, os.PathLike]) -> [str, os.PathLike]:
    return unidecode(re.sub('\\s{1,}', '_', src))


def absoluteFilePaths(src: Union[str, os.PathLike]) -> list:
    """
    Return all absolute file paths in ``src`` recursively.

    Source:
    https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
    """

    for dpath, _, fnames in os.walk(src):
        for f in fnames:
            yield os.path.abspath(os.path.join(dpath, f))


def flatten(nested_seq: Union[Iterable, Sequence]) -> list:
    """
    Return vectorized (1D) list from nested Sequence ``nested_seq``.
    """

    return [bottomElem for sublist in nested_seq for bottomElem
            in (flatten(sublist)
                if (isinstance(sublist, Sequence)
                    and not isinstance(sublist, str))
                else [sublist])]


def sizeof_fmt(num: int, suffix: str = 'B') -> str:
    """
    Return the size of the input in a humanly readable manner.

    Source:
    https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
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
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty}


def megamerge(dflist: list, howto: str = "outer",
              onto: str = None) -> object:
    """
    Returns a pd.DataFrame made from merging the frames in ``dflist``.
    """

    return reduce(lambda x, y: pd.merge(
        x, y, on=onto, how=howto).astype("object"), dflist)
