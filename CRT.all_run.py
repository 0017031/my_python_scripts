#!/usr/bin/env python
# coding: utf-8
import importlib
import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass, astuple, asdict
from pathlib import Path
from typing import Iterable, Final, List, Tuple

from colorama import Fore, init as colorama_init
colorama_init(autoreset=True)


@dataclass(frozen=True)
class InputConfigPaths:
    mrconfig: Path
    verifier_metadata: Path
    FilteredEvents: Path


@dataclass
class OutputNames:
    verifier: str = "verifier_output.log"
    model: str = "model_output.log"
    stdout: str = "standard_output.log"
    stderr: str = "error_out.log"
    model_recording: str = "model_recording.log"
    srf_txt: str = "srf.txt"


def get_root_dir():  # Where pgmodel.exe lies under.
    if len(sys.argv) == 1:
        return os.getcwd()  # Running under current folder: python this.py
    else:
        return sys.argv[1]  # Or, set by argv[1]:           python this.py w:\PGModel_Win10-VS2015_migration\Res2_20.06.12


def validate_paths(paths: Iterable[Path], resolve=False):
    for p in paths:
        assert p.exists(), f"Can't find {Fore.RED}{p}"
        if not resolve:
            print(f"{Fore.GREEN}{p}")
        else:
            print(f"{Fore.GREEN}{p}")
            print(f"{Fore.GREEN}{p.resolve()}")
    print()


def input_paths(slog_source_dir_: Path) -> (Path, Path, [Path], InputConfigPaths):
    root_dir_ = Path(get_root_dir())
    pgmodel_ = root_dir_ / r"Systems\Design\Model\pgmodel.exe"

    verify_py: Path = root_dir_ / r"CommonTools\PGTAF\MAOV\plugin\Verify.py"
    pgmodel_dir: Path = pgmodel_.parent
    slog_list: [Path] = list(slog_source_dir_.rglob("*.log"))
    config_files = InputConfigPaths(
        mrconfig=root_dir_ / r"Systems\Design\Model\Model_Recording\mrconfig.dat",
        verifier_metadata=root_dir_ / r"Systems\Design\Architecture\verifier_metadata.dat",
        FilteredEvents=root_dir_ / r"Systems\Design\Architecture\FilteredEvents.dat")

    print(f"Running pgmodel.exe with verify.py in folder: {Fore.BLUE}{root_dir_}\n")

    validate_paths([slog_source_dir_, verify_py, pgmodel_, pgmodel_dir / r"model.ini"])
    validate_paths(astuple(config_files))

    # check the sort-log/merge-log/convert-tool that are required by verify.py
    sys.path.insert(1, str(verify_py.parent))
    validate_paths(Path(getattr(importlib.import_module(verify_py.stem), name))
                   for name in ('sortLogDirectory', 'mergeLogDirectory', 'modelRecordConvertDirectory'))

    return verify_py, pgmodel_dir, slog_list, config_files


def output_paths(input_slog: Path, vlog_dir_name: str) -> (Path, OutputNames):
    out_dir = Path.cwd() / input_slog.stem[2:]  # s_Heart_Rate_Trend_Core.log -> Heart_Rate_Trend_Core
    vlog_dir_ = Path.cwd() / vlog_dir_name

    os.makedirs(vlog_dir_.as_posix(), exist_ok=True)
    os.makedirs(out_dir.as_posix(), exist_ok=True)

    vlog_path = vlog_dir_ / f"v_{input_slog.stem[2:]}.log"  # s_Heart_Rate_Trend_Core.log -> v_Heart_Rate_Trend_Core.log

    out_logs = OutputNames()
    for k, v in asdict(out_logs).items():
        setattr(out_logs, k, str(out_dir / v))  # prefix out_dir to each field

    return vlog_path, out_logs


def add_perl_path(dir_):
    if subprocess.run('where perl').returncode != 0:
        os.environ['PATH'] = f'{os.environ.get("PATH")};{dir_}'
        assert subprocess.run('where perl').returncode == 0, "Can't find perl.exe"


def run_one_verify_py(command: List[str], lock=None):  # -> Tuple[subprocess.CompletedProcess, float]:
    start0 = time.time()
    r = subprocess.run(list(command), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    t = time.time() - start0
    with lock:
        print(f'{Path(command[3]).stem} finished in {t:.6f} seconds')
    return r, t


def main():
    """:return total number of successful run"""
    verify_py, pgmodel_dir, slog_list, config = input_paths(s_log_source_dir)

    print(f'total {len(slog_list)} s-log files, running with {MAX_NUM_WORKER} worker-processes\n')
    # if environment variable "FULL" not defined, run only 1 s-log
    # slog_list = slog_list if 'FULL' in os.environ.keys() else slog_list[0:MAX_NUM_WORKER]

    my_command_list = []
    for slog in slog_list:
        vlog, out_logs = output_paths(slog, v_log_dir_name)
        my_command: [str] = list(map(str, ["python.exe", verify_py, pgmodel_dir, slog, "-m", vlog,
                                           # output, intermediate-logs
                                           "-vl", out_logs.verifier,
                                           "-model", out_logs.model,
                                           "-std", out_logs.stdout,
                                           "-e", out_logs.stderr,
                                           "-t", out_logs.model_recording,
                                           "-srf", out_logs.srf_txt,
                                           # input, config_files
                                           "-c", config.mrconfig,
                                           "-vm", config.verifier_metadata,
                                           "-filter", config.FilteredEvents]))
        my_command_list.append(my_command)

    s = time.time()
    with ThreadPoolExecutor(max_workers=MAX_NUM_WORKER) as pool:  # https://stackoverflow.com/a/40687012
        my_results: [Tuple[subprocess.CompletedProcess, float]] = \
            list(pool.map(lambda cmd: run_one_verify_py(cmd, multiprocessing.Manager().Lock()),
                          my_command_list))
    print(f'\nFinished {len(my_command_list)} logs in {(time.time() - s):.6f} seconds')

    my_std_out = [dict(vlog=Path(r.args[5]).name, return_code=r.returncode, stdout=r.stdout.decode('utf-8').splitlines())
                  for r, _ in my_results]

    with open('my_std_out.json', mode='w') as f2:
        json.dump(my_std_out, f2, indent=4)
    with open('my_results.dat', mode='wb') as f1:
        pickle.dump(my_results, f1)

    return len([r for r in my_results if r[0].returncode == 0])


# python this.py
# python this.py w:\PGModel_Win10-VS2015_migration\Res2_20.06.12
if __name__ == "__main__":
    s_log_source_dir: Final = Path(r"C:\Users\baic\temp\CRT\Ver_Log\slogs")
    v_log_dir_name: Final = "v_log_vs2015"
    MAX_NUM_WORKER = 10

    perl_dir: Final = Path(r"w:\perl_587_LDAP_with_excel\bin")
    add_perl_path(perl_dir)

    main()

# C:\python27\python.exe    %verify_PY%     %PGModel_exe_dir%   !input_s_log!  ^
# -vl %verifier_output% -model %model_output% -m !vlog_output! -std %standard_output% -e %error_output%  -t %model_recording_output% ^
# -c %input_mrconfig%  -vm %input_verifier_metadata% -filter %input_FilteredEvents%
