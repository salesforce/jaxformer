# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt

python3 1_split_raw.py 2>&1 | tee /tmp/dataset_v1/1_split_raw.out
'''
import sys
import os
import gzip
import json
import io
import random
import argparse
import glob

from smart_open import open

from util import print_time


def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.data_bucket_path = '/tmp/dataset_v1/0_raw/train.txt'
    
    args.out_bucket_path = '/tmp/dataset_v1/1_split_raw/{:012d}.txt'

    args.out_splits = 1024

    args.assert_samples_num = 1_000_000_000

    return args


def yield_lines(files):
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                yield line


def main():
    args = create_args()
    total_records = 0

    os.makedirs(os.path.dirname(args.out_bucket_path), exist_ok=True)

    with print_time('emunerating files'):
        files = [args.data_bucket_path]
        print(f'{len(files)} files', flush=True)

    with print_time(f'splitting {len(files)} files into {args.out_splits} files'):
        f_out_handles = [open(args.out_bucket_path.format(i), 'w') for i in range(args.out_splits)]
        for i, line in enumerate(yield_lines(files)):
            total_records += 1
            f_out_handles[i % len(f_out_handles)].write(f'{line}\n')

            if i % 100_000 == 0:
                print(i, args.assert_samples_num)

        for f_handle in f_out_handles:
            f_handle.close()

    print(f'total_records={total_records}', flush=True)

    assert total_records == args.assert_samples_num, f'{total_records} != {args.assert_samples_num}'

    print('done.', flush=True)


if __name__ == '__main__':
    main()