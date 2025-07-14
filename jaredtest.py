#!/usr/bin/env python3
import argparse
import subprocess
import sys
import re
from pathlib import Path

def parse_input_file(path):
    """
    Parse the input file for lines like:
    test/models/test_bert.py .                                               [  0%]
    Returns a dict mapping test names (without .py) to their status string.
    """
    tests = {}
    line_re = re.compile(r'^(?P<path>\S+\.py)\s+(?P<marks>[\.\wFxs]+)')
    with open(path) as f:
        for line in f:
            m = line_re.match(line)
            if not m:
                continue
            fullpath = m.group('path')
            name = Path(fullpath).stem  # e.g. test_bert
            marks = m.group('marks')
            tests[name] = marks
    return tests

def build_include_expr(names):
    """
    Build a pytest -k expression including only these names: "name1 and name2 and ..."
    """
    return ' and '.join(sorted(names))

def build_exclude_expr(names):
    """
    Build a pytest -k expression excluding these names: "not name1 and not name2 and ..."
    """
    parts = [f'not {n}' for n in sorted(names)]
    return ' and '.join(parts)

def main():
    p = argparse.ArgumentParser(
        description="Run pytest under python3 -m pytest and tee output to file+stdout"
    )
    p.add_argument('--input', '-i', metavar='FILE',
                   help="parser input file (to pick fails/passes)")
    p.add_argument('--output', '-o', metavar='FILE', default='log.log',
                   help="where to tee stdout+stderr (default: log.log)")
    group = p.add_mutually_exclusive_group()
    group.add_argument('--fails', '-f', action='store_true',
                       help="run only the tests that failed (requires --input)")
    group.add_argument('--passes', '-p', action='store_true',
                       help="run only the tests that passed (requires --input)")
    args = p.parse_args()

    # handle the -f / -p requirements
    if (args.fails or args.passes) and not args.input:
        sys.exit("error: --fails/--passes requires --input FILE")

    # Build pytest command
    cmd = ['python3', '-m', 'pytest', 'test/']

    if args.input:
        tests = parse_input_file(args.input)
        if args.fails:
            # include only failing: those with 'F'
            failing = [name for name, marks in tests.items() if 'F' in marks]
            if failing:
                expr = build_include_expr(failing)
                cmd += ['-k', expr]
        elif args.passes:
            # include only passing: exclude those with 'F'
            non_failing = [name for name, marks in tests.items() if 'F' not in marks]
            if non_failing:
                expr = build_exclude_expr(non_failing)
                cmd += ['-k', expr]
        # else neither: no -k

    # run and tee
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_path = args.output
    with open(output_path, 'w') as logfile:
        for line in proc.stdout:
            # write to console
            sys.stdout.write(line)
            # write to log
            logfile.write(line)

    ret = proc.wait()
    sys.exit(ret)

if __name__ == '__main__':
    main()
