#!/usr/bin/env python3
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='')
    parser.add_argument('--dest', type=str, help='')
    args = parser.parse_args()
    path_dir = args.src
    dest_dir = args.dest
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    dump_list = []
    for directory_path, _, files in os.walk(path_dir):
        for file in files:
            if file.endswith('.dump'):
                dump_list.append(os.path.join(directory_path, file))
    print(len(dump_list))
    for dump in dump_list:
        # os.remove(dump)
        os.system(f'cp {dump} {dest_dir}')



