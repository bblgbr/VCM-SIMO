#!/usr/bin/env python3
"""
Compare feature dump from two runs

Features output from the VCM feature decoder are logged to a format containing
per-layer and per-channel mean and variance.

These are compared between the two and differences exceeding a threshold are reported.
"""
import argparse
import json
import os
import sys

import numpy as np

THRESHOLD = 0.001

VERBOSE = True

def read_json_file(json_file):
    src_data = []
    with open(json_file, 'r') as f:
        finished = False
        while not finished:
            line = f.readline()
            if line:
                src_data.append(json.loads(line))
            else:
                finished = True
    return src_data

def compare_files(src1, src2, json_file, threshold):
    src1_data = read_json_file(f'{src1}/{json_file}')
    src2_data = read_json_file(f'{src2}/{json_file}')

    assert len(src1_data) == len(src2_data), f'Length mismatch for {json_file}'

    failed = False

    worst_cases = []
    for idx, (frame_a, frame_b) in enumerate(zip(src1_data, src2_data)):
        assert len(frame_a) == len(frame_b), f'Layer count mismatch in {json_file} at entry {idx}'

        for layer_idx, (layer_a, layer_b) in enumerate(zip(frame_a, frame_b)):
            for item in ['means', 'variances']:
                assert len(layer_a[item]) == len(layer_b[item]), f'Logged feature count mismatch in {json_file} at entry {idx} layer_idx {layer_idx} in item {item}'
                worst_case = np.max(np.abs(np.array(layer_a[item]) - np.array(layer_b[item])))
                worst_cases.append(worst_case)

                if worst_case > threshold:
                    if VERBOSE:
                        print(f'Tolerance exceeded in {json_file} at frame {idx} layer {layer_idx} for {item} with {worst_case}')
                    failed = True

    worst = np.max(worst_cases)
    sys.stdout.write(f' Worst-case absolute difference: {worst}')
    hist = np.histogram(worst_cases)
    #print(hist)
    return failed

def compare_dirs(src1, src2, threshold):
    src1_files = os.listdir(src1)
    src2_files = os.listdir(src2)
    src1_json = [f for f in src1_files if f.endswith('.dump')]
    src2_json = [f for f in src2_files if f.endswith('.dump')]

    src1_set = set(src1_json)
    src2_set = set(src2_json)

    src1_extra = list(src1_set.difference(src2_set))
    src2_extra = list(src2_set.difference(src1_set))

    assert len(src1_extra) == 0, f'First source has extra files: {src1_extra}'
    assert len(src2_extra) == 0, f'Second source has extra files: {src2_extra}'

    print(f'Found {len(src1_json)} files')
    failed = False
    for json_file in src1_json:
        sys.stdout.write(f'Checking {json_file}')
        failed = failed or compare_files(src1, src2, json_file, threshold)
        sys.stdout.write('\n')


    assert not failed, f'Difference tolerance exceeded'

    print('All OK!')

def main():
    """ 
    For invoking as an application (e.g. as job submitted to farm)
    """

    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False
    )   

    parser.add_argument('--src1', type=str, help='First source directory')
    parser.add_argument('--src2', type=str, help='Second source directory')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help='Difference threshold before error')
    args = parser.parse_args()

    compare_dirs(os.path.expandvars(args.src1), os.path.expandvars(args.src2), args.threshold)

if __name__ == "__main__":
    main()

