#!/usr/bin/env python
# coding: utf-8
import errno
import filecmp
import fileinput
import os
import sys
from pathlib import Path

from colorama import Fore, Style, init as colorama_init
colorama_init()


def main():
    file_xs = get_valid_vcxproj()
    changed_xs = modify_vcxproj(file_xs)
    report_files(changed_xs, 'changed', Fore.YELLOW)
    print('Done!')


def modify_vcxproj(file_xs):
    with fileinput.input(files=file_xs, inplace=True, backup='.bak') as f:
        for line in f:
            new_line = line.replace("v142", "v140")  # vs2019:v142, vs2015:v140
            if "WindowsTargetPlatformVersion" in new_line:
                new_line = ""
            print(new_line, end="")

    bak_xs = [f.with_suffix('.vcxproj.bak') for f in file_xs]
    changed_xs = [f for f, f_bak in zip(file_xs, bak_xs) if not filecmp.cmp(f, f_bak)]
    unchanged_bak_xs = [f_bak for f, f_bak in zip(file_xs, bak_xs) if filecmp.cmp(f, f_bak)]  # True if equal

    for f in unchanged_bak_xs:
        os.remove(f)

    return changed_xs


def get_valid_vcxproj() -> [Path]:
    file_xs = tuple(get_valid_root().rglob("*.vcxproj"))
    if len(file_xs) == 0:
        print('No *.vcxproj found.\n\nBye bye ~~')
        sys.exit(0)
    else:
        pass
        report_files(file_xs, 'found')
    return file_xs


def get_valid_root() -> Path:
    root = Path().absolute() if len(sys.argv) == 1 else Path(sys.argv[1]).absolute()
    root = get_valid_path_(root)

    warn_running_default_in_current_dir(root) if len(sys.argv) == 1 else None
    print(f'Changing all vcxproj file to vs2015 in folder: {Fore.GREEN}{root}{Style.RESET_ALL}\n')

    return root


def get_valid_path_(p) -> Path:
    if not p.exists():
        print(f'{Fore.RED}{p}{Style.RESET_ALL} does NOT exists!!')
        sys.exit(errno.ENOENT)  # ENOENT: No such file or directory
    return p


def report_files(changed_xs, msg, color=''):
    print(f'{len(changed_xs)} files {msg}:')
    print('\n'.join(f'{color}{f.as_posix()}' for f in changed_xs)) if len(changed_xs) != 0 else None
    print(Style.RESET_ALL)


def warn_running_default_in_current_dir(my_root: Path):
    if my_root.resolve() == Path().resolve():
        print(f'Current Directory: {Fore.YELLOW}{my_root.absolute()}{Style.RESET_ALL}, ', end='')
        answer: str = input(f'run in this directory, (yes/No)? ')
        if len(answer) != 0 and answer[0].lower() == 'y':
            return
        else:
            print(f'Bye bye ~~')
            sys.exit(0)


if __name__ == '__main__':
    main()
