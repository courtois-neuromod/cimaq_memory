#!/usr/bin/python3

import inspect
import sys

def get_default_args(func: callable) -> dict:
    """
    Return a dict containing the default arguments of ``func``.

    Useful for allowing keyword arguments being passed to another function.

    Args:
        func: callable
            A callable function from which to retrive its default parameters.

    Returns: dict
        Dict containing a function's default parameters.

    Notes:
    https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """

    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty}

def main():
    get_default_args(sys.argv[1])

if __name__ == '__main__':
    main()

