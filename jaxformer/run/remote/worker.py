# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import time
import argparse
import traceback
import socket

import numpy as np

import jax

from jaxformer.utils import StoppableThread, print_time
from .protocol import *

from jaxformer.models.factory import create_model


def main(args):

    with print_time(f'Awaiting master on port {args.port}'):
        s_l = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s_l.bind(('', args.port))
        s_l.listen(5)
        s, addr = s_l.accept()
        print(f'Received handshake from master {addr}')

    with print_time(f'Awaiting config from master'):
        print('FN_INIT_CALL (begin)', flush=True)
        fn, args = socket_read(s)
        print('FN_INIT_CALL (end)', flush=True)
        assert fn == FN_INIT_CALL
        mesh_shape, config = args
        print(f'mesh = {mesh_shape}')
        print(f'config = {config}')

    with print_time(f'Allocating jax devices'):
        print(f'Jax.device_count() = {jax.device_count()}')
        devices = np.array(jax.devices()).reshape(mesh_shape)

    with jax.experimental.maps.Mesh(devices, ('dp', 'pt', 'mp')):
        with print_time(f'Model initialization'):
            (model, optimizer, lr_schedule), try_save_ckpt, load_ckpt = create_model(config=config)

        try:
            while True:
                step = int(model.state['step'])
                print(f'{step} AWAIT')

                fn, args = socket_read(s)

                if fn == FN_TRAIN_CALL:
                    print(f'{step} FN_TRAIN_CALL')
                    step, loss, grad_global_norm = model.train(args)
                    print(f'{step} FN_TRAIN_RET')
                    socket_write(s, FN_TRAIN_RET, [step, loss, float(lr_schedule(step)), grad_global_norm])

                elif fn == FN_PROFILE_CALL:
                    print(f'{step} FN_PROFILE_CALL')
                    model.profile(args)
                    print(f'{step} FN_PROFILE_CALL')
                    socket_write(s, FN_PROFILE_RET, [step])

                elif fn == FN_SAVE_CALL:
                    print(f'{step} FN_SAVE_CALL')

                    def keep_alive(thread):
                        attempt = 0
                        while not thread.stopped():
                            print(f'{step} FN_SAVE_WAIT_CALL', [attempt])
                            socket_write(s, FN_SAVE_WAIT_CALL, [attempt])
                            attempt += 1
                            time.sleep(10)
                        print(f'{step} keep-alive stopped')

                    thread = StoppableThread(do_run=keep_alive)
                    thread.start()
                    try_save_ckpt(model.state, args)
                    thread.stop()

                    process_count = int(jax.process_count())

                    print(f'{step} FN_SAVE_RET (begin)', [step, process_count])
                    socket_write(s, FN_SAVE_RET, [step, process_count])
                    print(f'{step} FN_SAVE_RET (end)')

                elif fn == FN_LOAD_CALL:
                    print(f'{step} FN_LOAD_CALL')

                    def keep_alive(thread):
                        attempt = 0
                        while not thread.stopped():
                            print(f'{step} FN_LOAD_WAIT_CALL', [attempt])
                            socket_write(s, FN_LOAD_WAIT_CALL, [attempt])
                            attempt += 1
                            time.sleep(10)
                        print(f'{step} keep-alive stopped')

                    thread = StoppableThread(do_run=keep_alive)
                    thread.start()
                    model.state = load_ckpt(model.state, *args)
                    thread.stop()

                    process_count = int(jax.process_count())

                    step = int(model.state['step'])
                    print(f'{step} FN_LOAD_RET (begin)', [step, process_count])
                    socket_write(s, FN_LOAD_RET, [step, process_count])
                    print(f'{step} FN_LOAD_RET (end)')

                elif fn == FN_STATS_CALL:
                    print(f'{step} FN_STATS_CALL')
                    stats = model.stats()
                    print(f'{step} FN_STATS_RET', stats)
                    socket_write(s, FN_STATS_RET, stats)
                    
                else:
                    raise Exception(f'Unknown remote function {fn}')
        finally:
            s.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    try:
        main(args=parse_args())
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)