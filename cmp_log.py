# -*- coding: utf-8 -*-
"""
    compare common-log-files.

    use "list_patterns" to filter out the lines you don't want to compare.
    use "glob_logfile" to define the file type
"""

import argparse
import errno
import logging
import re
import subprocess
import sys
import time
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from difflib import unified_diff
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Pattern, Iterable, Collection, Dict
from colorama import Fore, Style, init as colorama_init
colorama_init(autoreset=True)

# global variables
glob_logfile = '*.log'
result_lines: Pattern[str] = re.compile(f'[+][^+]|[-][^-]')  # udiff result lines are prefixed either by '-' or '+'
prefix_length = 1
external_diff_exe = Path(r'f:\git\usr\bin\diff.exe')

# (Events|Configuration) File|ation Date & Time|StdVersion\|ModelLibrary
list_patterns = [
    'Configuration File',
    'Events File',
    'ation Date & Time',
    'MODEL|StdVersion|ModelLibrary']
# create the filter object: escape the original separators "|" in the string, then concatenate all the re-expressions.
version_lines: Pattern[str] = re.compile('|'.join(re.escape(e) for e in list_patterns))


# noinspection DuplicatedCode
def compare_one_with_hash(old_file: Path, new_file: Path) -> [[str], int]:  # using set {}
    # def hash_my_string(s):
    #     return adler32(s.encode('utf-8'))
    #     return hashlib.sha1(s.encode())  # https://stackoverflow.com/a/3546075/3353857

    def prefix(p: str, xs: Iterable[str]):
        return list(f'{p}{x}' for x in xs)

    def remove_version_line(xs: Iterable[str]) -> Iterable[str]:
        return filter(lambda x: not version_lines.search(x),
                      xs)

    with open(old_file.as_posix(), 'r') as fh_old, open(new_file.as_posix(), 'r') as fh_new:
        xs_old = fh_old.readlines()
        xs_new = fh_new.readlines()

    hashed_set_new = set(map(hash, xs_new))
    hashed_set_old = set(map(hash, xs_old))

    only_in_old_file = prefix('-', (remove_version_line(line for line in xs_old if hash(line) not in hashed_set_new)))
    only_in_new_file = prefix('+', (remove_version_line(line for line in xs_new if hash(line) not in hashed_set_old)))

    diff = tuple(chain(only_in_old_file, only_in_new_file))
    count = max(len(only_in_old_file), len(only_in_new_file))

    return diff, count


# noinspection DuplicatedCode
def compare_one_external_diff(f1: Path, f2: Path, diff=external_diff_exe) -> [str]:
    """diff, sort, then diff again"""
    command = (diff.as_posix(), f1.as_posix(), f2.as_posix(), r'--unified=0')  # diff.exe old.log new.log --unified=0
    r = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    # from diff result filter our lines with leading '+' or '-', and also ignore "version" lines
    diff = tuple(filter(lambda x: not version_lines.search(x) and result_lines.match(x),
                        r.stdout.splitlines(keepends=True)))

    old = tuple(line[prefix_length:] for line in diff if line.startswith('-'))
    new = tuple(line[prefix_length:] for line in diff if line.startswith('+'))

    # for the above result, sort them , and compare again
    diff = tuple(filter(lambda x: result_lines.match(x),
                        unified_diff(sorted(old), sorted(new))))
    count = int(len(diff) / 2 + .5)

    return diff, count


# noinspection DuplicatedCode
def compare_logs_sequencial(file_name_xs: [str], old_files_dict: Dict[str, Path], new_files_dict: Dict[str, Path]):  # use name as the key to locate path in the dict
    count_xs = []
    for i, file in enumerate(file_name_xs, start=1):
        diff, count = compare_one_with_hash(old_files_dict[file], new_files_dict[file])
        report_result(diff, old_files_dict[file], new_files_dict[file], n_current=i, n_total_files=len(file_name_xs))
        count_xs.append(count)

    total_diff_count = sum(count_xs)
    good_file_n = sum(1 for n in count_xs if n == 0)
    bad__file_n = sum(1 for n in count_xs if n != 0)

    if len(file_name_xs) > 1:
        logging.info(f'\n***\ntotal {total_diff_count} differences in {len(file_name_xs)} files, {good_file_n} same, {bad__file_n} diff')


# noinspection DuplicatedCode
def compare_logs_parallel(file_name_xs: [str], old_files_dict: Dict[str, Path], new_files_dict: Dict[str, Path]):  # use name as the key to locate path in the dict
    W = 10

    # print(f'compare_logs_parallel, thread, external-diff , ~~~ workers:{W}')

    def my_compare_fn_wrapper(t: [Path, Path]) -> [[str], int]:
        return compare_one_with_hash(t[0], t[1])

    # file_name_xs = file_name_xs[:1]
    count_xs = []
    with ThreadPoolExecutor(max_workers=W) as pool:
        future_xs = {pool.submit(compare_one_external_diff, old_files_dict[f], new_files_dict[f]): f
                     for f in file_name_xs}
        for i, future in enumerate(as_completed(future_xs), start=1):
            file = future_xs[future]
            diff, count = future.result()
            count_xs.append(count)
            report_result(diff, old_files_dict[file], new_files_dict[file], n_current=i, n_total_files=len(file_name_xs))

        # result_xs = list(pool.map(my_compare_fn_wrapper, xs))
        # for r, file, i in zip(result_xs, file_name_xs, range(1, len(file_name_xs))):
        #     diff, count = r
        #     report_result(diff, old_files_dict[file], new_files_dict[file], n_current=i, n_total_files=len(file_name_xs))
        #     count_xs.append(count)

    total_diff_count = sum(count_xs)
    good_file_n = sum(1 for n in count_xs if n == 0)
    bad__file_n = sum(1 for n in count_xs if n != 0)

    if len(file_name_xs) > 1:
        logging.info(f'\n***\ntotal {total_diff_count} differences in {len(file_name_xs)} files, {good_file_n} same, {bad__file_n} diff')


def files_from_dir(dir_old, dir_new) -> [[str], Dict[str, Path], Dict[str, Path]]:
    # use Dict to recover Path from lower-case names
    old_files: Dict[str, Path] = {path.name.lower(): path for path in dir_old.glob(glob_logfile)}
    new_files: Dict[str, Path] = {path.name.lower(): path for path in dir_new.glob(glob_logfile)}

    unique_old_xs_ = old_files.keys() - new_files.keys()
    unique_new_xs_ = new_files.keys() - old_files.keys()
    common_xs: [str] = sorted(new_files.keys() & old_files.keys())
    count_ = len(common_xs)

    report_file_status(dir_old, unique_old_xs_, 'unique')
    report_file_status(dir_new, unique_new_xs_, 'unique')
    logging.info(f"****\n{count_} common-files in both directories\n")

    return common_xs, old_files, new_files


def files_from_list(d1: Path, d2: Path, file_list: Collection[str]) -> [[str], Dict[str, Path], Dict[str, Path]]:
    not_in_d1_xs_: [str] = [f for f in file_list if not Path(d1 / f).exists()]
    not_in_d2_xs_: [str] = [f for f in file_list if not Path(d2 / f).exists()]

    common_xs: [str] = [n for n in file_list if Path(d1 / n).exists() and Path(d2 / n).exists()]
    old_files: Dict[str, Path] = {n: Path(d1 / n) for n in file_list if Path(d1 / n).exists()}
    new_files: Dict[str, Path] = {n: Path(d2 / n) for n in file_list if Path(d2 / n).exists()}

    report_file_status(d1, not_in_d1_xs_, 'MISSING')
    report_file_status(d2, not_in_d2_xs_, 'MISSING')
    logging.info(f"****\n{len(common_xs)} common-files in both directories\n")

    return common_xs, old_files, new_files


def report_result(diff: [str], old_file: Path, new_file: Path, n_current=0, n_total_files=0):
    count = int(len(diff) / 2 + 0.5)

    progress_str = f'{n_current}/{n_total_files}' if n_total_files > 1 else ''
    count_str = f'{count:^6} diffs' if count != 0 else 'same(0 diff)'
    file_name_str = f'{old_file.name}' if n_total_files > 1 else ''

    logging.info(f'{progress_str:7} {count_str:12}, {file_name_str}')

    verbose_logger = logging.info if n_total_files < 5 else logging.debug
    if count != 0:
        verbose_logger(f'--- : {old_file}')
        verbose_logger(f'+++ : {new_file}')
        verbose_logger(''.join(diff))


def report_file_status(d: Path, xs_files: Collection[str], status: str):
    if len(xs_files) != 0:
        logging.info(f"{len(xs_files)} {status} files in directory: ({d})")
        # logging.info('\n\t'.join(sorted(xs_files)) + '\n')
    if len(xs_files) < 6:
        logging.info('\n'.join(map(lambda x: f'\t{x}', sorted(xs_files))) + '\n')
    else:
        logging.info('\n'.join(sorted([f'\t{i}, {f}' for i, f in enumerate(xs_files, start=1)])) + '\n')


def count_line_wc(f: Path) -> [Path, int]:
    my_command = [r'C:\Users\baic\Downloads\app\gnuwin32_orig1\bin\wc.exe', '-l', f]
    result = subprocess.run(my_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n_line = int(result.stdout.split()[0])
    return f, n_line


def count_dir_with_reduce(dir_name: Path, trace=True):
    def sum_and_trace(x: int, yy: [Path, int]):
        print(f'{yy[1]:10,} lines, {yy[0]}')
        return x + yy[1]

    file_list = tuple(dir_name.rglob(glob_logfile))

    t0 = time.time()
    with ThreadPoolExecutor() as pool:
        n_total_lines = reduce(sum_and_trace,
                               pool.map(count_line_wc, file_list), 0)

    print(f'\ntotal {n_total_lines:,} lines in {len(file_list)} files (wc.exe ran in {time.time() - t0:.3f} seconds)')
    return


def count_dir_with_for_loop(dir_name: Path, trace=True):
    file_list = tuple(dir_name.rglob(glob_logfile))

    n_total_lines, t0 = 0, time.time()
    with ThreadPoolExecutor() as pool:
        for f, n in pool.map(count_line_wc, file_list):
            n_total_lines += n
            print(f'{n:10,} lines, {f}') if trace else None
    print(f'\ntotal {n_total_lines:,} lines in {len(file_list)} files (wc.exe ran in {time.time() - t0:.3f} seconds)')


def time_it(func, n=1):  # run func n times and time it.
    f1, f2 = [Path(s) for s in
              # (r"ba.log", r"bb.log")]
              (r"old.log", r"new.log")]
    n = n if n > 1 else 1

    t0 = time.time()
    ret = [func(f1, f2) for _ in range(n)]
    t1 = time.time() - t0

    mul, unit = [1, 's'] if t1 > 1 else [1000, 'ms']
    print(f'{t1 * mul / n:.3f} {unit}, {func.__name__},\t\t{n} round(s) in {t1 * mul:.3f} {unit}')
    ret = sorted(list(ret[0]))
    return ret

    # c = 1
    # if len(sys.argv) > 1 and sys.argv[1].isdigit():
    #     c = int(sys.argv[1])
    #
    # ss0 = time.time()
    # executor = ProcessPoolExecutor
    # # executor = ThreadPoolExecutor
    # with executor(max_workers=4) as pool:
    #     rs = list(pool.map(time_it,
    #                        [compare_log_file, compare_log_file_external_diff]))
    # print(f'ttal {time.time() - ss0}seconds')
    # sys.exit()


def setup_logger(diff_record_file='diff_summary.txt', my_format='%(message)s'):
    """ :return: None """

    # diff_summary.txt: DEBUG and above
    logging.basicConfig(level=logging.DEBUG, datefmt='%m-%d %H:%M', format=my_format,
                        filename=diff_record_file, filemode='w')

    # diff_summary_brief.txt: INFO and above
    fh_brief = logging.FileHandler(f'{Path(diff_record_file).stem}_brief.txt', mode='w')
    fh_brief.setLevel(logging.INFO)
    fh_brief.setFormatter(logging.Formatter(my_format))  # '%(message)s'  # remove 'DEBUG:root' or 'INFO:root:' from the log

    # stderr: INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(my_format))

    logging.getLogger().addHandler(fh_brief)  # getLogger() returns the root logger when name=None
    logging.getLogger().addHandler(console)


def check_arguments(args, parser_):
    for p in [args.f1, args.f2, args.file_list]:
        if p is not None and not Path(p).exists():
            print(f"\n! Please check your input. Can't open {Fore.RED}{p}\n")
            parser_.print_usage(),
            sys.exit(errno.ENOENT)

    if Path(args.f1).is_file() != Path(args.f2).is_file():
        f = f'{Fore.LIGHTRED_EX}{args.f1 if Path(args.f1).is_file() else args.f2}{Style.RESET_ALL}'
        d = f'{Fore.LIGHTCYAN_EX}{args.f1 if Path(args.f1).is_dir() else args.f2}{Style.RESET_ALL}'
        print(f"\n! Can't compare a directory {d} to a file {f}.\n")
        parser_.print_usage()
        sys.exit(errno.EINVAL)


def parse_arg_and_check_path():
    parser = argparse.ArgumentParser(description='Compare common-log-files', formatter_class=argparse.RawTextHelpFormatter)  # noqa
    parser.add_argument('f1', help='the 1st(left)  file/dir to compare')
    parser.add_argument('f2', help='the 2nd(right) file/dir to compare')
    parser.add_argument('-O', '--output', help='file to store the diff summary, default "diff_summary.txt"', default='diff_summary.txt')
    parser.add_argument('-L', '--file_list', help='compare only files in the list, in both dir1 and dir2 (one name perl line)')
    my_args = parser.parse_args()

    check_arguments(my_args, parser)

    ll: [str] = None
    if my_args.file_list is not None:
        with open(my_args.file_list) as f:
            ll = f.read().splitlines()
    return Path(my_args.f1), Path(my_args.f2), ll, Path(my_args.output)


def my_main(p1: Path, p2: Path, file_list: [str] = None):  # p1,p2 are both valid, file_list maybe None
    if p1.is_file() and p2.is_file():
        name = p1.name
        compare_logs_parallel(file_name_xs=[name], old_files_dict={name: p1}, new_files_dict={name: p2})

    if p1.is_dir() and p2.is_dir():
        if file_list is None:
            name_xs, dict1, dict2 = files_from_dir(p1, p2)
        else:
            name_xs, dict1, dict2 = files_from_list(p1, p2, file_list)

        compare_logs_parallel(file_name_xs=name_xs, old_files_dict=dict1, new_files_dict=dict2)


# python cmp_log.py dir1  dir2   -O diff_summary.txt
# python cmp_log.py file1 file2  --output=diff_summary.txt
# python cmp_log.py file1 file2  --file_list=fileList.txt --output=diff_summary.txt
if __name__ == "__main__":
    assert external_diff_exe.exists(), f"Can't find {Fore.RED}{external_diff_exe}"

    if len(sys.argv) == 1:
        count_dir_with_for_loop(Path())

    path1, path2, compare_list_if_not_none, output = \
        parse_arg_and_check_path()

    setup_logger(output)
    logging.info('\nRunning:\n' + ' '.join(sys.argv) + '\n')

    start = time.perf_counter()
    my_main(path1, path2, compare_list_if_not_none)
    end = time.perf_counter()

    logging.info(f'\nfinished in {(end - start):.3f} seconds')
    sys.exit(0)

