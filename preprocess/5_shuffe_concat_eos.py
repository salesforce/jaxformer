# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

python3 5_shuffe_concat_eos.py 2>&1 | tee /tmp/dataset_v1/5_shuffe_concat_eos.out

curl https://sdk.cloud.google.com | bash
gcloud init
gsutil -m cp -r /tmp/data/dataset_v1/5_shuffe_concat_eos gs://sfr-tpu-us-east1-research/enijkamp/dataset_v1
'''
import os
import random
import argparse
from glob import glob
from multiprocessing import Pool, cpu_count
from time import time
from functools import partial

import tensorflow as tf


def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.data_bucket_path = '/tmp/dataset_v1/4_create_tf_records'

    # TODO(enijkamp): set eos id for tokenizer
    args.out_eos_value = 50256

    args.out_seq_len = 2048
    args.out_seq_len_min = 64

    args.out_records_per_file = int(2**16)
    args.out_bucket_path = '/tmp/dataset_v1/5_shuffe_concat_eos'

    args.n_workers = safe_cpu_count() // 2

    return args


def safe_cpu_count(default=32):
    try:
        return cpu_count()
    except NotImplementedError:
        return default


def yield_ds(args, files):
    def parse_fn(sample):
        parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
        return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

    def append_bos_fn(sample):
        return tf.pad(sample, paddings=[[1, 0]], mode='CONSTANT', constant_values=args.out_bos_value)

    def append_eos_fn(sample):
        return tf.pad(sample, paddings=[[0, 1]], mode='CONSTANT', constant_values=args.out_eos_value)

    # ds = tf.data.TFRecordDataset(files, num_parallel_reads=len(files))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=args.n_workers)
    ds = ds.shuffle(buffer_size=1024*64, seed=args.seed)
    ds = ds.map(parse_fn)
    # ds = ds.map(append_bos_fn)
    ds = ds.map(append_eos_fn)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for sample in ds:
        yield sample.numpy().tolist()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    feature = { 'text': _int64_feature(data) }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def process_files(args):
    print(f'listing files', flush=True)

    files = [f for f in glob(f'{args.data_bucket_path}/*')]
    random.seed(args.seed)
    random.shuffle(files)

    print(f'splitting {len(files)} files into {args.n_workers} partitions', flush=True)

    files_chunks = split(files, args.n_workers)

    print(f'processing {len(files_chunks)} partitions in parallel', flush=True)

    bind_process_files_chunk = partial(process_files_chunk, args)

    t0 = time()
    pool = Pool(processes=args.n_workers)
    pool.starmap(bind_process_files_chunk, [(i, file_chunk) for (i, file_chunk) in enumerate(files_chunks)])
    t1 = time()
    print('time', t1-t0, len(files_chunks))


def process_files_chunk(args, i, files):

    print(i, f'processing {len(files)} files', flush=True)

    def split(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    def yield_cut_samples(ds):
        for sample in ds:
            yield split(sample, args.out_seq_len)

    ds = yield_ds(args, files)

    seqs_to_prepend = []
    seqs_to_write = []

    n_total_records = 0
    n_out_files = 0

    for samples_cut in yield_cut_samples(ds):

        # progress
        if n_total_records % 100000 == 0:
            print(i, n_total_records, n_out_files)

        n_total_records += 1

        # store incomplete remainder sequence
        seq_len = len(samples_cut[-1])
        if seq_len < args.out_seq_len:
            seq_remainder = samples_cut.pop(-1)
            if seq_len >= args.out_seq_len_min:
                seqs_to_prepend.extend(seq_remainder)

        # append completed remainder sequences
        if len(seqs_to_prepend) >= args.out_seq_len:
            seqs_to_write.append(seqs_to_prepend[:args.out_seq_len])
            seqs_to_prepend = seqs_to_prepend[args.out_seq_len:]

        # append completed sequences
        seqs_to_write.extend(samples_cut)

        # write chunk
        if len(seqs_to_write) >= args.out_records_per_file:

            out_file = f'{args.out_bucket_path}/data_{i:04d}_{n_out_files:04d}.tfrecords'
            out_file_tmp = f'{out_file}.tmp'
            os.makedirs(os.path.dirname(out_file_tmp), exist_ok=True)

            print(i, f'writing file {out_file_tmp}', flush=True)
            n_records = 0

            with tf.io.TFRecordWriter(out_file_tmp, options=tf.io.TFRecordOptions(compression_method=None, compression_type='')) as writer:
                for record in seqs_to_write:
                    write_to_file(writer, record)
                    n_records += 1

            print(i, f'finalizing file {out_file_tmp} -> {out_file} with {n_records} records', flush=True)
            os.rename(out_file_tmp, out_file)

            n_out_files += 1
            seqs_to_write = []

    # final chunk
    if len(seqs_to_write) >= 100:

        out_file = f'{args.out_bucket_path}/data_{i:04d}_{n_out_files:04d}.tfrecords'
        out_file_tmp = f'{out_file}.tmp'
        os.makedirs(os.path.dirname(out_file_tmp), exist_ok=True)

        print(i, f'writing file {out_file_tmp}', flush=True)

        with tf.io.TFRecordWriter(out_file_tmp, options=tf.io.TFRecordOptions(compression_method=None, compression_type='')) as writer:
            for record in seqs_to_write:
                write_to_file(writer, record)

        print(i, f'finalizing file {out_file_tmp} -> {out_file} with {n_records} records', flush=True)
        os.rename(out_file_tmp, out_file)

    print(i, 'done', n_out_files)


def main():
    args = create_args()
    process_files(args)
    print('done.', flush=True)


if __name__ == '__main__':
    main()
