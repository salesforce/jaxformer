# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import time
import socket
import json

import numpy as np

from func_timeout import func_set_timeout

from smart_open import open

from jaxformer.utils import print_time, par_map

from .protocol import *
from .tpu import spawn_tpu_workers


class RemoteMaster:
    @func_set_timeout(1200)
    def __init__(self,
                 worker_addresses,
                 mesh_shape,
                 config):

        def connect_to_workers():

            def try_connect(address, attempts=10, wait_secs=10):
                for i in range(attempts):
                    try:
                        print(f'Connection attempt {i} of {attempts} to worker {address}')
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        s.connect(address)
                        return s
                    except Exception as e:
                        print(e)
                        time.sleep(wait_secs)
                raise Exception(f'Can not connect to worker {address} within {attempts} attempts')

            with print_time(f'Connecting to {len(worker_addresses)} workers'):
                worker_sockets = []
                for i, (worker_ip, worker_port) in enumerate(worker_addresses):
                    print(f'Connecting to worker {i} at {worker_ip}:{worker_port}')
                    s = try_connect((worker_ip, worker_port))

                    socket_write(s, FN_INIT_CALL, [mesh_shape, config])

                    print(f'Connected to worker {i} at {worker_ip}:{worker_port}')

                    # TODO(enijkamp): FN_INIT_CALL is not received without sleep, this should be needed
                    time.sleep(1)
    
                    worker_sockets.append(s)

            return worker_sockets

        self.worker_sockets = connect_to_workers()


    @func_set_timeout(600)
    def train(self, data):
        data_chunks = np.array_split(data, len(self.worker_sockets), axis=1)

        for s, d in zip(self.worker_sockets, data_chunks):
            socket_write(s, FN_TRAIN_CALL, {'x': d[:, :, :-1], 'y': d[:, :, 1:]})

        workers_return = [socket_read(s)[1] for s in self.worker_sockets]

        return_steps = np.array([worker_return[0] for worker_return in workers_return])
        return_loss = np.array([worker_return[1] for worker_return in workers_return])
        return_lr = np.array([worker_return[2] for worker_return in workers_return])
        return_grad = np.array([worker_return[3] for worker_return in workers_return])

        assert np.all(return_steps == return_steps[0]), return_steps

        return int(return_steps[0]), return_loss.mean(), return_lr[0], return_grad[0]


    @func_set_timeout(600)
    def profile(self, data):
        data_chunks = np.array_split(data, len(self.worker_sockets), axis=1)

        for s, d in zip(self.worker_sockets, data_chunks):
            socket_write(s, FN_PROFILE_CALL, {'x': d[:, :, :-1], 'y': d[:, :, 1:]})

        workers_return = [socket_read(s)[1] for s in self.worker_sockets]

        return_steps = np.array([worker_return[0] for worker_return in workers_return])

        assert np.all(return_steps == return_steps[0]), return_steps

        return int(return_steps[0])


    @func_set_timeout(2000)
    def save(self, step, path, wandb_run_id, data_files, data_file, data_batch):

        for s in self.worker_sockets:
            socket_write(s, FN_SAVE_CALL, path)

        print('socket_read', 'wait', 'begin')
        def await_save(s):
            worker_data_fn, workers_return = socket_read(s)
            while worker_data_fn == FN_SAVE_WAIT_CALL:
                worker_data_fn, workers_return = socket_read(s)
            assert worker_data_fn == FN_SAVE_RET
            return workers_return
        workers_return = par_map(await_save, self.worker_sockets)
        print('socket_read', 'wait', 'end')

        workers_steps = np.array([worker_return[0] for worker_return in workers_return])
        workers_process_counts = np.array([worker_return[1] for worker_return in workers_return])

        print(f'workers_steps={workers_steps}')
        print(f'workers_process_counts={workers_process_counts}')

        workers_step = workers_steps[0]
        process_count = workers_process_counts[0]

        assert step == workers_step

        assert np.all(workers_steps == workers_step), workers_steps
        assert np.all(workers_process_counts == process_count), workers_process_counts

        with print_time(f'Writing ckpt json at step={step}'):
            with open(f'{path}/ckpt.json', 'w') as f:
                json.dump({'process_count': int(process_count), 'step': int(step), 'wandb_run_id': wandb_run_id, 'data_files': data_files, 'data_file': data_file, 'data_batch': data_batch}, f)

        return step


    # TODO(enijkamp): for pjit partition, we save each partition. For pjit xmap emulation, we only need to save global state once.
    @func_set_timeout(2000)
    def save_xmap(self, step, path):

        s = self.worker_sockets[0]
        print('socket_write', 'begin')
        socket_write(s, FN_SAVE_CALL, path)
        print('socket_write', 'end')

        print('socket_read', 'wait', 'begin')
        worker_data_fn, workers_return = socket_read(s)
        while worker_data_fn == FN_SAVE_WAIT_CALL:
            print(worker_data_fn, workers_return)
            worker_data_fn, workers_return = socket_read(s)
        print('socket_read', 'wait', 'end')

        worker_return = workers_return

        workers_steps = worker_return[0]
        workers_process_counts = worker_return[1]

        print(f'workers_steps={workers_steps}')
        print(f'workers_process_counts={workers_process_counts}')

        workers_step = workers_steps
        process_count = workers_process_counts

        assert step == workers_step

        assert workers_steps == workers_step, workers_steps
        assert workers_process_counts == process_count, workers_process_counts

        with print_time(f'Writing ckpt json at step={step}'):
            with open(f'{path}/ckpt.json', 'w') as f:
                json.dump({'process_count': int(process_count), 'step': int(step)}, f)


    @func_set_timeout(3000)
    def load(self, path, step=None, ignore_optimizer=False):
        for s in self.worker_sockets:
            socket_write(s, FN_LOAD_CALL, [path, step, ignore_optimizer])

        print('socket_read', 'wait', 'begin')
        def await_load(s):
            worker_data_fn, workers_return = socket_read(s)
            while worker_data_fn == FN_LOAD_WAIT_CALL:
                print(worker_data_fn, workers_return)
                worker_data_fn, workers_return = socket_read(s)
            assert worker_data_fn == FN_LOAD_RET
            return workers_return
        workers_return = par_map(await_load, self.worker_sockets)
        print('socket_read', 'wait', 'end')

        workers_steps = np.array([worker_return[0] for worker_return in workers_return])
        workers_process_counts = np.array([worker_return[1] for worker_return in workers_return])

        print(f'workers_steps={workers_steps}')
        print(f'workers_process_counts={workers_process_counts}')

        workers_step = workers_steps[0]
        process_count = workers_process_counts[0]

        assert step == workers_step

        assert np.all(workers_steps == workers_step), workers_steps
        assert np.all(workers_process_counts == process_count), workers_process_counts

        with print_time(f'Loading ckpt json at from {path}/ckpt.json'):
            with open(f'{path}/ckpt.json', 'r') as f:
                ckpt = json.load(f)
                print('loaded checkpoint config')
                print(ckpt)
                return ckpt


    @func_set_timeout(600)
    def stats(self):
        for s in self.worker_sockets:
            socket_write(s, FN_STATS_CALL, {})

        workers_0_stats = np.array([socket_read(s)[1] for s in self.worker_sockets])[0]
        print(f'workers_0_stats={workers_0_stats}')
        return workers_0_stats


def create_master(config):

    with print_time(f'Spawning TPU'):
        endpoints_ips = spawn_tpu_workers(tpu_user=config['tpu_user'],
            tpu_spawn=config['tpu_spawn'],
            tpu_create_env=config['tpu_create_env'],
            tpu_name=config['tpu_name'],
            tpu_tags=config['tpu_tags'],
            tpu_image=config['tpu_image'],
            tpu_zone=config['tpu_zone'],
            tpu_version=config['tpu_version'],
            tpu_size=config['tpu_size'],
            tpu_network=config['tpu_network'],
            tpu_subnetwork=config['tpu_subnetwork'],
            tpu_worker_port=config['tpu_worker_port'],
            tpu_delete=config['tpu_delete'],
            tpu_reserved=config['tpu_reserved'],
            tpu_internal_ips=config['tpu_internal_ips'])


    with print_time(f'Creating local worker'):
        pt = config['opt_params_partitions']
        dp = config['tpu_size_logical'] // config['tpu_cores'] // config['opt_params_partitions']
        mp = config['tpu_cores']
        mesh_shape = (dp, pt, mp)
        print(f'mesh_shape={mesh_shape}')
        master = RemoteMaster(endpoints_ips, mesh_shape, config)

    return master