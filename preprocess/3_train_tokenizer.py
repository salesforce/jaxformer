# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

python3 2_train_tokenizer.py 2>&1 | tee /tmp/dataset_v1/2_train_tokenizer.out
'''
import os
import argparse

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers, processors, trainers)

from util import print_time


def create_args(args = argparse.Namespace()):

    args.tokenizer_min_frequency = 1
    args.tokenizer_vocab = 'ABCDEFGHIKLMNOPQRSTUVWXYZ12'
    args.tokenizer_vocab_size = len(args.tokenizer_vocab) + 3 # 27 chars + <bos> + <eos> + <pad> = 30 tokens

    args.tokenizer_file = '/tmp/dataset_v1/tokenizer.json'

    return args


def create_data_iter(args):
    yield args.tokenizer_vocab


def train(args, data_iter):
    with print_time('initialize tokeniser'):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    with print_time('train bpe'):
        trainer = trainers.BpeTrainer(vocab_size=args.tokenizer_vocab_size, min_frequency=args.tokenizer_min_frequency, special_tokens=['<|pad|>', '<|bos|>', '<|eos|>'])
        tokenizer.train_from_iterator(data_iter, trainer)

    with print_time('serialize bpe'):
        os.makedirs(os.path.dirname(args.tokenizer_file), exist_ok=True)
        tokenizer.save(args.tokenizer_file, pretty=True)

    with print_time('check'):
        vocab_chars = [c for c in args.tokenizer_vocab] + ['<|pad|>', '<|bos|>', '<|eos|>']
        assert len(vocab_chars) == 30, len(vocab_chars)
        for c1 in tokenizer.get_vocab():
            assert c1 in vocab_chars

    with print_time('assert'):
        assert len(args.tokenizer_vocab) == 27
        print(tokenizer.get_vocab())
        print(len(tokenizer.get_vocab()))
        assert tokenizer.get_vocab_size() == 30, tokenizer.get_vocab_size()
        


def main():
    args = create_args()
    train(args, create_data_iter(args))
    print('done.')


if __name__ == '__main__':
    main()
