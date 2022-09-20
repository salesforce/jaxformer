# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf


def list_files(path):
    is_gcs_path = path.startswith('gs://')
    filenames = tf.io.gfile.glob(path) if is_gcs_path else [str(p) for p in Path(path).glob(path)]
    return sorted(filenames)


def create_iterator_from_tfrecords_files(filenames, seq_len, batch_size, skip=0):

    def parse_fn(sample):
        parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
        return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

    def truncate_fn(sample):
        return sample[:seq_len]

    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.repeat()
    ds = ds.skip(skip)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(truncate_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=np.prod(batch_size), drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    for batch in ds:
        yield batch.numpy().reshape([batch_size[0], batch_size[1], seq_len])


def create_resumable_iter(files, batch_size, seq_len, resume_at_file=None, resume_at_batch=None):

    def iterator_from_tfrecords_files(files, seq_len, batch_size):

        def parse_fn(sample):
            parsed_features = tf.io.parse_single_example(sample, {'text': tf.io.VarLenFeature(tf.int64)})
            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

        def truncate_fn(sample):
            return sample[:seq_len]

        ds = tf.data.TFRecordDataset(files)
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(truncate_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size=np.prod(batch_size), drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for batch in ds:
            yield batch.numpy().reshape([batch_size[0], batch_size[1], seq_len])
    
    def resume_to_file(files_iter, resume_at_file=None):
        f = next(files_iter)
        print(f'Skipping from {f} to file {resume_at_file}')
        while f != resume_at_file:
            f = next(files_iter)
        print(f'Skipped to file {f}')
        assert f == resume_at_file
        return files_iter, f

    files_iter = iter(itertools.cycle(files))

    if resume_at_file is not None:
        files_iter, f = resume_to_file(files_iter, resume_at_file)
        batch_iter = iterator_from_tfrecords_files(f, seq_len=seq_len, batch_size=batch_size)
        for b, records in enumerate(batch_iter):
            if b < resume_at_batch:
                continue
            yield (records, f, b)

    for f in files_iter:
        batch_iter = iterator_from_tfrecords_files(f, seq_len=seq_len, batch_size=batch_size)
        for b, records in enumerate(batch_iter):
            yield (records, f, b)


def create_mock_iter(vocab_size, batch_size, seq_len):
    b = 0
    while True:
        yield (np.random.randint(vocab_size, size=[batch_size[0], batch_size[1], seq_len]), 'file', b)
        b += 1


def test_restore_state(files, seq_len):
    batch_size = (32, 16)
    n_batches = 32

    loader_train_iter = create_iterator_from_tfrecords_files(files, seq_len=seq_len, batch_size=batch_size)
    for _ in range(n_batches):
        next(loader_train_iter)
    check_sample = next(loader_train_iter)

    loader_train_iter = create_iterator_from_tfrecords_files(files, seq_len=seq_len, batch_size=batch_size, skip=n_batches*np.prod(batch_size))
    check_sample2 = next(loader_train_iter)

    assert tf.reduce_sum(tf.cast(tf.not_equal(check_sample, check_sample2), tf.uint32)).numpy() == 0


def load_records_np(files, seq_len):
    batch_size = (32, 16)
    samples_iter = create_iterator_from_tfrecords_files(files, seq_len=seq_len, batch_size=batch_size)
    sample = next(samples_iter)
    print(sample.shape)
    assert sample.shape[0:2] == batch_size
    assert sample.shape[2] == seq_len


def test_data_iterator(files, seq_len):
    load_records_np(files=files, seq_len=seq_len)
    test_restore_state(files=files, seq_len=seq_len)