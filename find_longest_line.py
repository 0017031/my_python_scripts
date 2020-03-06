from __future__ import print_function

import argparse
import errno
import os
import subprocess
import sys
from pathlib import Path
from colorama import Fore, Back
import colorama
colorama.init(autoreset=True)


def find_long_lines_old(f: Path):
    with open(f.as_posix()) as file:
        the_index, the_length, the_line = (0, 0, '')
        for i, (i_len, i_line) in enumerate((len(line), line) for line in file):
            if i_len > the_length:
                the_index, the_length, the_line = i, i_len, i_line

        console_width = os.get_terminal_size().columns - 5
        ending = '...' if len(the_line) > console_width else ''

        print(f'File: {Fore.GREEN}{f}')
        print(f'- total {i + 1:,} lines , longest at #{the_index + 1:,}, length {the_length:,}')
        print(f'{Back.CYAN}{the_line[:console_width]}{ending}')


def find_long_lines_m1(f: Path):
    with open(f.as_posix()) as my_file:
        line_xs = tuple(my_file)

    the_line = max(line_xs, key=len)
    the_index = line_xs.index(the_line)
    # the_index, the_line = max(enumerate(my_lines) , key=lambda x: len(x[1])) # enumerate is 15% slower

    console_width = os.get_terminal_size().columns - 5
    ending = '...' if len(the_line) > console_width else ''

    print(f'File: {Fore.GREEN}{f}')
    print(f'- total {len(line_xs):,} lines , longest at #{the_index + 1:,}, length {len(the_line):,}')
    print(f'{Back.CYAN}{the_line[:console_width]}{ending}')


def find_long_lines_m2(f: Path):  # wc then max(), 20% slower

    def wc_count(file):
        out = subprocess.run(['wc', '-l', file], stdout=subprocess.PIPE, text=True).stdout
        return int(out.split()[0])

    wc: int = wc_count(f)

    with open(os.path.abspath(f)) as my_file:
        the_index, the_line = max(enumerate(my_file), key=lambda x: len(x[1]))  # enumerate is 15% slower

    console_width = os.get_terminal_size().columns - 5
    ending = ' ...' if len(the_line) > console_width else ''

    print(f'File: {Fore.GREEN}{f}')
    print(f'- total {wc:,} lines , longest at #{the_index + 1:,}, length {len(the_line):,}')
    print(f'{Back.CYAN}{the_line[:console_width]}{ending}')


def verify_path(f: Path):
    if not f.exists():
        print(f"{Fore.YELLOW}!!! Please check your input. Can't open {Fore.RED}{f}\n")
        parser.print_usage(),
        sys.exit(errno.ENOENT)  # ENOENT: No such file or directory


# python find_longest_line.py log1.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='* Open a logfile, find the longest line of it')
    parser.add_argument('file', help='the logfile to open and parse')
    args = parser.parse_args()
    verify_path(Path(args.file))

    find_long_lines_m1(Path(args.file))
