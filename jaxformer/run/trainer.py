# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import time
from datetime import datetime
import random
from contextlib import nullcontext

import numpy as np

from jaxformer.utils import print_time
from jaxformer.data.iterator_resumable import list_files, create_resumable_iter, create_mock_iter, test_data_iterator

from .local.master import create_master as create_master_local
from .remote.master import create_master as create_master_remote


def create_master(config):
    if config['debug_run_local']:
        return create_master_local(config)
    else:
        return create_master_remote(config)


def set_default_config(config):

    config['tpu_version'] = config.get('tpu_version', 3)
    config['tpu_cores'] = {3: 8, 4: 4}[config['tpu_version']]
    config['tpu_size_logical'] = {3: config['tpu_size'], 4: config['tpu_size'] // 2}[config['tpu_version']]
    config['tpu_name'] = config['tpu_name'].format(**config)
    config['tpu_network'] = config.get('tpu_network', None)
    config['tpu_subnetwork'] = config.get('tpu_subnetwork', None)
    config['tpu_tags'] = config.get('tpu_tags', None)
    config['tpu_reserved'] = config.get('tpu_reserved', False)
    config['tpu_internal_ips'] = config.get('tpu_internal_ips', False)
    
    config['opt_gradient_accumulation_steps'] = config.get('opt_gradient_accumulation_steps', 1)

    config['restore_reinit'] = config.get('restore_reinit', False)

    config['restore_ckpt'] = config.get('restore_ckpt', None)
    config['restore_step'] = config.get('restore_step', 0)
    config['restore_retry'] = config.get('restore_retry', 0)

    config['wandb_run_id'] = config.get('wandb_run_id', None)

    assert config['tpu_version'] in [3, 4]
    assert config['tpu_cores'] == {3: 8, 4: 4}[config['tpu_version']]
    assert config['tpu_size'] in [8, 32, 64, 128, 256, 512, 1024]
    assert config['model_vocab_size'] % config['tpu_cores'] == 0

    return config


def train(config):

    with print_time(f'Check configuration'):
        config = set_default_config(config)

        tpu_size_logical = config['tpu_size_logical']
        tpu_cores = config['tpu_cores']

        opt_gradient_accumulation_steps = config['opt_gradient_accumulation_steps']
        opt_per_replica_batch = config['opt_per_replica_batch']

        model_vocab_size = config['model_vocab_size']
        model_seq_len = config['model_seq_len']

        ckpt_dir = config['ckpt_dir']

        restore_reinit = config['restore_reinit']
        restore_ckpt = config['restore_ckpt']
        restore_step = config['restore_step']


    with print_time(f'Creating master'):
        master = create_master(config)


    if restore_ckpt is not None:
        with print_time(f'Restoring checkpoint at {restore_ckpt}/{restore_step}'):
            if restore_reinit:
                print(f'Re-initializing optimizer')
                ckpt_config = master.load(path=f'{restore_ckpt}/{restore_step}', step=0, ignore_optimizer=True)
                ckpt_step = 0
            else:
                ckpt_config = master.load(path=f'{restore_ckpt}/{restore_step}', step=restore_step)
                ckpt_step = int(ckpt_config['step'])
                assert restore_step == ckpt_step
            print(f'Continuing training at step {ckpt_step}')


    if config['debug_mock_data']:

        with print_time('Mocking dataset iterator'):
            print(f'Total tokens per step = {opt_gradient_accumulation_steps * opt_per_replica_batch * (tpu_size_logical // tpu_cores) * model_seq_len}')
            batch_size = (opt_gradient_accumulation_steps, opt_per_replica_batch * tpu_size_logical // tpu_cores)
            loader_train_iter = create_mock_iter(model_vocab_size, batch_size, model_seq_len)
            train_files = ['file']

    else:

        with print_time('Loading dataset iterator'):
            print(f'Total tokens per step = {opt_gradient_accumulation_steps * opt_per_replica_batch * (tpu_size_logical // tpu_cores) * model_seq_len}')
            batch_size = (opt_gradient_accumulation_steps, opt_per_replica_batch * tpu_size_logical // tpu_cores)
            if restore_reinit:
                train_files = list_files(config['data_train_set'])
                random.shuffle(train_files)
                test_data_iterator(train_files, model_seq_len)
                loader_train_iter = create_resumable_iter(train_files, batch_size, model_seq_len, resume_at_file=None, resume_at_batch=None)
            else:
                if restore_ckpt is None:
                    train_files = list_files(config['data_train_set'])
                    random.shuffle(train_files)
                    test_data_iterator(train_files, model_seq_len)
                    loader_train_iter = create_resumable_iter(train_files, batch_size, model_seq_len, resume_at_file=None, resume_at_batch=None)
                else:
                    train_files = ckpt_config['data_files']
                    loader_train_iter = create_resumable_iter(train_files, batch_size, model_seq_len, resume_at_file=ckpt_config['data_file'], resume_at_batch=ckpt_config['data_batch'])

            print(train_files)


    with print_time('Loading first batch'):
        (data, data_file, data_batch) = next(loader_train_iter)
        assert data.shape == (batch_size[0], batch_size[1], model_seq_len)


    with print_time('Compiling train function'):
        step = master.train(data)[0]


    with print_time('Gathering stats'):
        stats = master.stats()
        print(stats)
        config = {**config, **stats}


    if config['wandb_enabled']:
        with print_time('Initializing wandb'):
            import wandb
            run = wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], group=config['wandb_group'], name=f"{config['wandb_name']}.{config['ckpt_attempt']}", config=config, reinit=True)
            wandb_run_id = run.id
            print(f'wandb_run_id={wandb_run_id}')
    else:
        run = nullcontext()
        wandb_run_id = None


    if restore_reinit or restore_ckpt is None:
        with print_time(f'Saving checkpoint at step {step}'):
            ckpt_step = master.save(step=step, path=f'{ckpt_dir}/{step}', wandb_run_id=wandb_run_id, data_files=train_files, data_file=data_file, data_batch=data_batch)


    with run:
        while step < int(config['opt_total_steps']):

            t0 = time.time()
            (data, data_file, data_batch) = next(loader_train_iter)
            assert data.shape == (batch_size[0], batch_size[1], model_seq_len)
            step, loss, lr, grad_global_norm = master.train(data)
            t1 = time.time()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            samples_sec = np.prod(data.shape[0:2]) / (t1 - t0)
            tokens_sec = np.prod(data.shape[0:3]) / (t1 - t0)
            steps_sec = 1. / (t1 - t0)

            if config['wandb_enabled']:
                wandb.log({'train/loss': loss, 'train/lr': lr, 'train/grad_global_norm': grad_global_norm, 'perf/samples_sec': samples_sec, 'perf/tokens_sec': tokens_sec, 'perf/steps_sec': steps_sec}, step)
            print(f'{timestamp} step={step:08d} lr={lr:.7f} loss={loss:.3f} data_file={data_file} data_batch={data_batch} grad_global_norm={grad_global_norm:.3f} data.shape={data.shape} steps/sec={steps_sec:.3f} samples/sec={samples_sec:.3f} tokens/sec={tokens_sec:.0f}')

            if step > 0 and step % config['ckpt_every'] == 0 or step == int(config['opt_total_steps']):
                with print_time(f'Saving checkpoint at step {step}'):
                    ckpt_step = master.save(step=step, path=f'{ckpt_dir}/{step}', wandb_run_id=wandb_run_id, data_files=train_files, data_file=data_file, data_batch=data_batch)



def profile(config):

    with print_time(f'Check configuration'):
        config = set_default_config(config)

        tpu_size_logical = config['tpu_size_logical']
        tpu_cores = config['tpu_cores']

        opt_gradient_accumulation_steps = config['opt_gradient_accumulation_steps']
        opt_per_replica_batch = config['opt_per_replica_batch']

        model_vocab_size = config['model_vocab_size']
        model_seq_len = config['model_seq_len']


    with print_time(f'Creating master'):
        master = create_master(config)


    if config['debug_mock_data']:

        with print_time('Mocking dataset iterator'):
            print(f'Total tokens per step = {opt_gradient_accumulation_steps * opt_per_replica_batch * (tpu_size_logical // tpu_cores) * model_seq_len}')
            batch_size = (opt_gradient_accumulation_steps, opt_per_replica_batch * tpu_size_logical // tpu_cores)
            loader_train_iter = create_mock_iter(model_vocab_size, batch_size, model_seq_len)
            train_files = ['file']

    else:

        with print_time('Loading dataset iterator'):
            print(f'Total tokens per step = {opt_gradient_accumulation_steps * opt_per_replica_batch * (tpu_size_logical // tpu_cores) * model_seq_len}')
            batch_size = (opt_gradient_accumulation_steps, opt_per_replica_batch * tpu_size_logical // tpu_cores)

            train_files = list_files(config['data_train_set'])
            random.shuffle(train_files)
            test_data_iterator(train_files, model_seq_len)
            loader_train_iter = create_resumable_iter(train_files, batch_size, model_seq_len, resume_at_file=None, resume_at_batch=None)

            print(train_files)


    with print_time('Loading first batch'):
        (data, data_file, data_batch) = next(loader_train_iter)
        assert data.shape == (batch_size[0], batch_size[1], model_seq_len)


    with print_time('Compile train function'):
        step = master.train(data)[0]


    with print_time('Profile train function'):
        step = master.profile(data)