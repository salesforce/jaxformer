# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev

python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0 tensorflow-cpu==2.5.0

pip install -e .

deepspeed --num_gpus=1 jaxformer/hf/train.py
'''

########################################################################################################
## imports

import os
import argparse
import random

from time import time

import numpy as np

import torch

from transformers import AutoConfig, AutoModelForCausalLM

import deepspeed

from jaxformer.utils import print_time
from jaxformer.data.iterator_resumable import list_files, create_resumable_iter, create_mock_iter, test_data_iterator


########################################################################################################
## args

DEEPSPEED_CONFIG = \
{
    'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 12, 'hysteresis': 2, 'min_loss_scale': 1},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-05, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}},
    'scheduler': {'type': 'WarmupLR', 'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100}},
    'zero_optimization': {
        'stage': 3,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_param_persistence_threshold': 40960,
        'stage3_max_live_parameters': 1e9,
        'stage3_max_reuse_distance': 1e9,
        'stage3_gather_fp16_weights_on_model_save': True
    },
    'train_batch_size': 32,
    'train_micro_batch_size_per_gpu': 2,
    'gradient_accumulation_steps': 16,
    'gradient_clipping': 1.0,
    'steps_per_print': 8,
    'wall_clock_breakdown': False,
    'compression_training': {'weight_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {}, 'different_groups': {}}}
}


def create_args(args=argparse.Namespace()):

    args.seed = 42

    args.model = 'Salesforce/codegen-16B-mono'

    args.deepspeed_config = DEEPSPEED_CONFIG

    args.data_train_set = 'gs://sfr-tpu-us-east1-research/enijkamp/jaxformer/datasets/thepile/train/*.tfrecords'

    args.opt_steps_train = 1000

    args.model_seq_len = 2048
    args.model_vocab_size = 51200

    args.debug_fixed_batch = False
    args.debug_mock_data = False

    return args



########################################################################################################
## train

def train(args):

    #######################
    ## preamble

    set_seed(args.seed)


    #######################
    ## data

    if args.debug_mock_data:

        with print_time('Mocking dataset iterator'):
            print(f"Total tokens per step = {args.deepspeed_config['train_micro_batch_size_per_gpu'] * args.model_seq_len}")
            batch_size = [1, args.deepspeed_config['train_micro_batch_size_per_gpu']]
            loader_train_iter = create_mock_iter(args.model_vocab_size, batch_size, args.model_seq_len)
            train_files = ['file']

    else:

        with print_time('Loading dataset iterator'):
            print(f"Total tokens per step = {args.deepspeed_config['train_micro_batch_size_per_gpu'] * args.model_seq_len}")
            batch_size = [1, args.deepspeed_config['train_micro_batch_size_per_gpu']]
            train_files = list_files(args.data_train_set)
            random.shuffle(train_files)
            test_data_iterator(train_files, args.model_seq_len)
            loader_train_iter = create_resumable_iter(train_files, batch_size, args.model_seq_len, resume_at_file=None, resume_at_batch=None)
            print(train_files)



    #######################
    ## model

    with print_time('Initializing model'):

        config = AutoConfig.from_pretrained(args.model)
        config.gradient_checkpointing = True
        config.use_cache = False

        model = AutoModelForCausalLM.from_pretrained(args.model, config=config)

        model.train()
        # TODO(enijkamp): we need to set this flag twice?
        model.gradient_checkpointing_enable()


    #######################
    ## deepspeed

    with print_time('Initializing deepspeed'):

        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        model_engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model, model_parameters=model_parameters)

        torch.cuda.empty_cache()


    #######################
    ## train

    print('Starting training')

    input_ids_fixed = torch.randint(low=0, high=args.model_vocab_size, size=[args.deepspeed_config['train_micro_batch_size_per_gpu'], args.model_seq_len], dtype=torch.int64).cuda()

    for step in range(args.opt_steps_train+1):

        # data

        if args.debug_fixed_batch:
            input_ids = input_ids_fixed
        else:
            (data, data_file, data_batch) = next(loader_train_iter)
            input_ids = torch.tensor(data[0].astype(np.int32)).long().cuda()

        assert input_ids.shape == torch.Size([args.deepspeed_config['train_micro_batch_size_per_gpu'], args.model_seq_len])

        # gradient

        loss = model_engine(input_ids=input_ids, labels=input_ids).loss

        model_engine.backward(loss)
        model_engine.step()


        # stats

        print(f'{step} {loss:8.3f}')



########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))




########################################################################################################
## main

def main():

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # args
    args = create_args()
    args.output_dir = output_dir
    args.exp_id = exp_id

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)

    # train
    train(args=args)



if __name__ == '__main__':
    main()