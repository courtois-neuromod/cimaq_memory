#!usr/bin/env/python3

from collections.abc import Iterable

def evenodd(inpt: Iterable) -> tuple:
    """ Returns a tuple containg (even items, odd items) from 'inpt' 
    ----------------------------------------------------------------
    Parameter(s)
    ------------
    inpt: any type of iterable sequence with any level/depth of nesting
    """
    eveseq = tuple(ele[1] for ele in enumerate(inpt) if ele[0] % 2 == 0)
    oddseq = tuple(ele[1] for ele in enumerate(inpt) if ele[0] % 2 != 0)
    return (eveseq, oddseq)

def main():
    if __name__ == __main__:
        evenodd(inpt)

