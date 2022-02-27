#!/bin/usr/python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def sizeof_fmt(num, suffix='B'):
    '''
    Return the size of the input in a humanly readable manner
    Source: https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    '''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def main():
    init_dict = dict(prog=sizeof_fmt,
                     usage=sizeof_fmt.__doc__,
                     description=__name__.__doc__.splitlines()[0],
                     formatter_class=ArgumentDefaultsHelpFormatter)

    _parser = ArgumentParser(**init_dict)
    _parser.add_argument('num', type=int,
                         help='Total number of bytes', nargs=1)
    _parser.add_argument('-s', '--suffix',
                         dest='suffix', nargs='?', default='B')
    args = _parser.parse_args()
    sizeof_fmt(args.num[0],args.suffix)

if __name__ == '__main__':
    main()
