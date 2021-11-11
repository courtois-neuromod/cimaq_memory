#!usr/bin/env/python3

from argparse import ArgumentParser
from collections.abc import Iterable
from typing import Sequence
from typing import Union

def flatten(
    nested_seq: Union[Iterable, Sequence],
    prnt: bool = False
) -> list:
    """
    Return unidimensional list from nested list using list comprehension.

    Args:
        nestedlst: Iterable, Sequence
            Nested sequence containing other lists, sequences etc.
        prnt: bool (default = False)
            Print output to stdout.

    Variables:
        bottomElem: type = str
        sublist: type = list
    ------
    Returns: list
        flatlst: unidimensional list
    """

    return [
        bottomElem
        for sublist in nested_seq
        for bottomElem in (
            flatten(sublist)
            if (isinstance(sublist, Sequence) and \
                not isinstance(sublist, str))
            else [sublist]
        )
    ]

def main():
    parser = ArgumentParser(prog=flatten, usage=flatten.__doc__,
                            description=flatten.__doc__.splitlines()[1])
    parser.add_argument('-p', '--print', dest='prnt', action='store_true')
    parser.add_argument('nested_seq', nargs='+')
    args = parser.parse_args()
    flatten(args.nested_seq, args.prnt)

if __name__ == '__main__':
     main()
