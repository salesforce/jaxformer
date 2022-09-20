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

python3 2_dedup_raw.py 2>&1 | tee /tmp/dataset_v1/2_dedup_raw.out
'''
import os
import random
import argparse
import pickle
from multiprocessing.pool import ThreadPool
from glob import glob
import hashlib

from util import print_time

hashes = set()

def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.data_bucket_path = '/tmp/dataset_v1/1_split_raw'

    args.out_max_duplicates = 100
    args.out_bucket_path = '/tmp/dataset_v1/2_dedup_raw'

    args.n_threads = 64

    return args


def sha256str(s):
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def keep_file(x):
    h = sha256str(x)

    new_hash = not h in hashes
    if new_hash:
        hashes.add(h)

    return new_hash


def process_file(args, file):
    
    print(f'reading file {file}', flush=True)

    total_records_in = 0
    total_records_out = 0

    random.seed(args.seed)

    def yield_samples(file):
        with open(file, 'r') as f:
            for line in f:
                yield line

    out_file = f'{args.out_bucket_path}/{os.path.basename(file)}'
    out_file_tmp = f'{out_file}.tmp'
    os.makedirs(os.path.dirname(out_file_tmp), exist_ok=True)

    if os.path.exists(out_file):
        print(f'skipping file {out_file}', flush=True)
        return total_records_in, total_records_out

    with open(out_file_tmp, 'w') as f:
        for i, record in enumerate(yield_samples(file)):
            if i % 10_000 == 0:
                print(file, i, flush=True)
            total_records_in += 1
            if keep_file(record):
                total_records_out += 1
                f.write(record)

    print(f'finalizing file {out_file_tmp} -> {out_file} with {total_records_out} records', flush=True)
    os.rename(out_file_tmp, out_file)
    print(f'finalized file {out_file_tmp}')

    print(f'finished total_records_in={total_records_in} total_records_out={total_records_out}')

    return total_records_in, total_records_out


def main():
    args = create_args()
    total_records_in, total_records_out = 0, 0

    with print_time('emunerating files'):
        files = [f for f in glob(f'{args.data_bucket_path}/*')]
        print(f'{len(files)} files', flush=True)

    with print_time(f'hashing files with {args.n_threads} threads'):
        with ThreadPool(args.n_threads) as pool:
            for (total_records_in_lang, total_records_out_lang) in pool.starmap(process_file, list(map(lambda l: (args, l), files))):
                total_records_in += total_records_in_lang
                total_records_out += total_records_out_lang

    with print_time(f'serializing {len(hashes)} hashes'):
        print(f'{len(hashes)} hashes', flush=True)
        with open(f'./2_dedup_raw_hashes.pickle', 'wb') as f:
            pickle.dump(hashes, f, protocol=pickle.HIGHEST_PROTOCOL)


    print(f'total_records_in={total_records_in} total_records_out={total_records_out}', flush=True)
    print('done.', flush=True)


if __name__ == '__main__':
    main()