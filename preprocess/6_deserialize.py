# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
'''
import argparse

import tensorflow as tf

from tokenizers import Tokenizer

from smart_open import open

from util import print_time


def create_args(args=argparse.Namespace()):

    args.alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ12'
    # args.data_in = '/tmp/data_protein_v2/5_shuffe_concat_eos/*'
    args.data_in = 'gs://sfr-tpu-prem2-bucket/enijkamp/data_protein_v3/5_shuffe_concat_eos/data_000000000000.tfrecords'
    args.tokenizer_file = 'gs://sfr-tpu-us-east1-research/enijkamp/protein/data_protein_v2/tokenizer.json'
    # args.ds_prefetch = 8

    return args


def load_records(args):

    with print_time('load iterator'):

        def tf_parse(example_proto):
            features = { 'text': tf.io.VarLenFeature(tf.int64) }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features['text'])), tf.uint32)

        ds = tf.data.TFRecordDataset(args.data_in)
        ds = ds.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
        # ds = ds.prefetch(args.ds_prefetch)


    with print_time('deseralize record'):

        # tokenizer = Tokenizer.from_file(args.tokenizer_file)
        with open(args.tokenizer_file, 'r') as f:
            tokenizer = Tokenizer.from_str(f.read())

        def to_iter(f):
            for x in f:
                yield x

        sample = next(to_iter(ds))
        sample_str = tokenizer.decode(sample.numpy(), skip_special_tokens=False)
        print(sample_str)
        print(sample.numpy().shape[0])

        def contains_eos(sample):
            array = sample.numpy()
            for i in range(len(array)):
                if array[i] == 2:
                    return True
            return False

        assert contains_eos(sample)


def main():
    args = create_args()
    load_records(args)
    print('done.')


if __name__ == '__main__':
    main()
