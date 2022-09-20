# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import io
import json

import jax
import jax.numpy as jnp

import numpy as np

from smart_open import open

from jaxformer.utils import print_time


# TODO(enijkamp): for inter-board sharding, we only need to serialize one copy
def save_ckpt(state, path):

    def save_arrays(arrays, fname):
        with open(fname, 'wb') as f:
            np.savez(f, *arrays)

    with print_time(f'Saving model in {path}'):
        save_arrays(jax.tree_flatten(state['model'])[0], f'{path}/model/{jax.process_index()}.npz')

    with print_time(f'Saving opt in {path}'):
        save_arrays(jax.tree_flatten(state['opt'])[0],  f'{path}/opt/{jax.process_index()}.npz')

    return int(state['step']), int(jax.process_count())


def try_save_ckpt(state, path, attempts=3):
    for i in range(attempts):
        try:
            print(f'Saving ckpt in {path} (attempt {i})')
            return save_ckpt(state, path)
        except:
            print(f'Failed saving ckpt in {path} (attempt {i})')
    raise Exception(f'Failed to save ckpt in {path} after {attempts} attempts')


# TODO(enijkamp): for inter-board sharding, we only need to de-serialize one copy
def load_ckpt(state_old, path, step_overwrite=None, ignore_optimizer=False):

    def load_arrays(old, fname):
        old_vals, treedef = jax.tree_flatten(old)
        with open(fname, 'rb') as f:
            loaded = np.load(io.BytesIO(f.read()))

        new_vals = [loaded[i] for i in loaded]

        for n, o in zip(new_vals, old_vals):
            assert o.shape == n.shape, f'Incompatible shapes {o.shape}!={n.shape}'

            if n.dtype == np.dtype('V2'):
                n.dtype = jnp.bfloat16

        return jax.tree_unflatten(treedef, new_vals)


    with print_time(f'Loading ckpt json from {path}'):
        with open(f'{path}/ckpt.json', 'r') as f:
            ckpt = json.load(f)


    assert jax.process_count() == ckpt['process_count']

    state_new = {
        'step': ckpt['step'] if step_overwrite is None else step_overwrite,
    }

    with print_time(f'Loading model from {path}'):
        state_new['model'] = load_arrays(state_old['model'], f'{path}/model/{jax.process_index()}.npz')

    if ignore_optimizer:
        state_new['opt'] = state_old['opt']
    else:
        with print_time(f'Loading opt from {path}'):
            state_new['opt'] = load_arrays(state_old['opt'], f'{path}/opt/{jax.process_index()}.npz')

    return state_new
