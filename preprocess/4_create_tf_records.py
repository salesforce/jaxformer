# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

export TRANSFORMERS_OFFLINE=1
python3 4_create_tf_records.py 2>&1 | tee /tmp/dataset_v1/4_create_tf_records.out
'''
import os
import argparse
import concurrent.futures

import transformers
from tokenizers import Tokenizer

import tensorflow as tf


def create_args(args=argparse.Namespace()):

    args.data_bucket_path = '/tmp/dataset_v1/2_dedup_raw'

    args.run_num_processes = 128

    args.out_bucket_path = '/tmp/dataset_v1/3_create_tf_records'

    # TODO(enijkamp): set custom tokenizer vocab
    args.tokenizer_custom = True
    args.tokenizer_file = '/tmp/dataset_v1/tokenizer.json'


    return args


# TODO(enijkamp): set custom tokenizer
def create_tokenizer(args):
    if args.tokenizer_custom:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
        encode = lambda sample: tokenizer.encode(sample).ids
        return encode
    else:
        transformers.GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        encode = lambda sample: tokenizer.encode(sample)
        return encode


def process_files(args):

    def process(files_chunks, do_map):

        def to_local_in_file(f):
            return f'{args.data_bucket_path}/{f}'

        def to_local_out_file(f):
            path = f'{args.out_bucket_path}/{f}'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return path

        total_num_records = 0
        total_num_files = 0
        for files_chunk in files_chunks:
            for f, (num_records, mean_len_record) in zip(files_chunk, do_map(process_file, [(args, to_local_in_file(f), to_local_out_file(f)) for f in files_chunk])):
                print(f, num_records, mean_len_record)
                print(f'file={f} mean_len_record={int(mean_len_record)} num_records={num_records} total_num_records={total_num_records}', flush=True)
                total_num_records += num_records
                total_num_files += 1
        print(f'finished {total_num_files} files and {total_num_records} total records', flush=True)

    def process_chunks(files_chunks):
        if args.run_num_processes == 1:
            def do_map(f, args_list):
                for args in args_list:
                    yield f(*args)
            process(files_chunks, do_map)
        else:
            with concurrent.futures.ProcessPoolExecutor(args.run_num_processes) as executor:
                def map_with_zip_args(f, args):
                    for result in executor.map(f, *zip(*args)):
                        yield result
                process(files_chunks, map_with_zip_args)

    def yield_chunks():
        files = os.listdir(f'{args.data_bucket_path}')
        for files_chunk in partition(l=files, n=args.run_num_processes):
            yield files_chunk

    print(f'processing', flush=True)
    process_chunks(yield_chunks())


def process_file(args, in_file, out_file):
    if os.path.exists(out_file):
        print(f'skipping {out_file}', flush=True)
        return 0, 0.

    def mv(f1, f2):
        os.rename(f1, f2)

    def rm(f):
        os.remove(f) if os.path.exists(f) else None

    out_file_tmp = f'{out_file}.tmp'
    rm(out_file_tmp)

    print(f'processing {in_file} -> {out_file_tmp}', flush=True)
    tokenizer = create_tokenizer(args)
    data_iter = create_fs_data_iter(in_file)
    n, mean_len = 0, 0.
    with tf.io.TFRecordWriter(out_file_tmp) as writer:
        for sample in data_iter:
            record = tokenizer(sample)
            write_to_file(writer, record)
            n += 1
            mean_len += (len(record) - mean_len) / n
            if n % 10_000 == 0:
                print(n, int(mean_len), in_file)

    print(f'finalizing {out_file_tmp} -> {out_file}', flush=True)
    mv(out_file_tmp, out_file)

    return n, mean_len



def create_fs_data_iter(in_file):
    with open(in_file, 'r') as f:
        for line in f:
            yield line


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    feature = { 'text': _int64_feature(data) }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def partition(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)] 


def main():
    # multiprocessing.set_start_method('spawn')
    args = create_args()
    
    process_files(args)

    print('done.', flush=True)


if __name__ == '__main__':
    main()