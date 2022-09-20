# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

### print ###

import time

class print_time():
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print(self.task)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.task} took {time.time()-self.t:.02f}s')


### sh ###

import selectors
import subprocess


class ShFail(Exception):
    def __init__(self, command):
        super().__init__(command)


def print_redirect(text, new_line=True):
    print(f'redirect: {text.rstrip()}')


def sh(command, new_line=True, check_return_code=True):
    return_code = run_loop(command=command, print=print_redirect, new_line=new_line)
    if check_return_code and return_code != 0:
        raise ShFail(command)


def sh_ret(command):
    return run_return(command, print=print_redirect)


def run_loop(command, print, new_line=False):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    sel = selectors.DefaultSelector()
    sel.register(process.stdout, selectors.EVENT_READ)
    sel.register(process.stderr, selectors.EVENT_READ)

    while True:
        for key, _ in sel.select():
            data = key.fileobj.readline()
            if not data:
                time.sleep(1)
                continue
            print(data, new_line=new_line)

        return_code = process.poll()
        if return_code is not None:
            print(f'return={return_code}')
            for output in process.stdout.readlines():
                print(output.strip(), new_line=new_line)
            for output in process.stderr.readlines():
                print(output.strip(), new_line=new_line)
            return return_code


def run_return(command, print):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return process.stdout.readline()


def run_return_code_old(command):
    import subprocess   
    result = subprocess.Popen(command, shell=True)
    output = result.communicate()[0]
    return result.returncode, output


def run_return_code(command):
    import subprocess   
    result = subprocess.run(command, shell=True)
    return result.returncode


def run(command, print):
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)



### threading ###

from multiprocessing.pool import ThreadPool

def par_map(f, args):
    return ThreadPool(processes=len(args)).map(f, args)


import threading

class StoppableThread(threading.Thread):
    def __init__(self, do_run,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_event = threading.Event()
        self.do_run = do_run

    def run(self):
        self.do_run(self)

    def stop(self):
        self.stop_event.set()
        self.join()

    def stopped(self):
        return self.stop_event.is_set()



### jax ###

import os
import jax

def emulate_tpu_on_cpu(cores=8):
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cores}'
    jax.config.update('jax_platform_name', 'cpu')
    return cores


def pjit_noop(fun,
         in_axis_resources,
         out_axis_resources,
         static_argnums = (),
         donate_argnums = ()):
    return fun


def with_sharding_constraint_noop(x, axis_resources):
    return x


### training ###

import jax
import jax.numpy as jnp
import optax

def create_lr_schedule_gpt3_fn(steps_warmup, steps_anneal, lr_max, lr_end):

    assert steps_warmup <= steps_anneal

    def lr_schedule_fn(step):

        bound = lambda x: jnp.clip(x, 0., 1.)

        lr_warmup = lambda step: lr_max * bound(step / steps_warmup)
        lr_anneal = lambda step: lr_max - (lr_max - lr_end) * (1 - jnp.cos(jnp.pi * bound((step - steps_warmup) / steps_anneal))) / 2

        lr = jax.lax.cond(step <= steps_warmup, lr_warmup, lr_anneal, step)

        return lr

    return lr_schedule_fn


def add_decayed_weights_exclude_ln_and_bias(weight_decay):
    mask_fn = lambda p: jax.tree_map(lambda x: x.ndim > 1, p)
    return optax.add_decayed_weights(weight_decay, mask_fn)


def to_f32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)


def to_f16(t):
    return jax.tree_map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, t)


def global_norm(updates):
    pre_sqrt = sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(updates)])
    return jnp.sqrt(pre_sqrt)