# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import jax

from jaxformer.utils import print_time, emulate_tpu_on_cpu
from jaxformer.models.factory import create_model

class LocalMaster:

    def __init__(self, mesh_shape, config):

        with print_time(f'Allocating jax devices'):
            print(f'Jax.device_count() = {jax.device_count()}')
            self.devices = np.array(jax.devices()).reshape(mesh_shape)

        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            model, optimizer, lr_schedule = create_model(config=config)[0]

        self.lr_schedule = lr_schedule
        self.model = model


    def train(self, data):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            step, loss, grad_global_norm = self.model.train({'x': data[:, :, :-1], 'y': data[:, :, 1:]})
            lr = float(self.lr_schedule(step))
            return step, loss, lr, grad_global_norm


    def profile(self, data):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            return self.model.profile({'x': data[:, :, :-1], 'y': data[:, :, 1:]})


    def save(self, step, path, wandb_run_id, data_files, data_file, data_batch):
        pass


    def load(self, path, step=None, ignore_optimizer=False):
        pass


    def stats(self):
        with jax.experimental.maps.Mesh(self.devices, ('dp', 'pt', 'mp')):
            return self.model.stats()


def create_master(config):

    tpu_size_logical = config['tpu_size_logical']
    tpu_cores = config['tpu_cores']
    rep = config['opt_params_partitions']

    if config['debug_emulate_tpu_on_cpu']:
        with print_time(f'Emulating tpu on cpu with {tpu_cores} cores'):
            emulate_tpu_on_cpu(cores=tpu_cores)


    with print_time(f'Creating local worker'):
        dp = tpu_size_logical // tpu_cores // rep
        mp = tpu_cores
        mesh_shape = (dp, rep, mp)
        print(f'mesh_shape={mesh_shape}')
        master = LocalMaster(mesh_shape, config)

    return master