# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import pickle


### protocol ###

FN_INIT_CALL = 0
FN_INIT_RET = 1

FN_TRAIN_CALL = 2
FN_TRAIN_RET = 3

FN_SAVE_CALL = 4
FN_SAVE_RET = 5

FN_LOAD_CALL = 6
FN_LOAD_RET = 7

FN_STATS_CALL = 8
FN_STATS_RET = 9

FN_SAVE_WAIT_CALL = 10
FN_SAVE_WAIT_RET = 11

FN_LOAD_WAIT_CALL = 12
FN_LOAD_WAIT_RET = 13

FN_PROFILE_CALL = 14
FN_PROFILE_RET = 15



### sockets ###

def socket_write(socket, fn, args):
    data_fn = int(fn).to_bytes(4, 'little', signed=False)
    data_bytes = pickle.dumps(args)
    data_size = len(data_bytes).to_bytes(8, 'little', signed=False)
    socket.send(data_fn)
    socket.send(data_size)
    socket.send(data_bytes)


def socket_read(s, buf_size=1024*32):
    data_fn_bytes = s.recv(4)
    data_fn = int.from_bytes(data_fn_bytes, 'little', signed=False)
    data_args_size_bytes = s.recv(8)
    data_args_size = int.from_bytes(data_args_size_bytes, 'little', signed=False)
    data_args_bytes = s.recv(min(data_args_size, buf_size))
    while len(data_args_bytes) != data_args_size:
        data_args_bytes += s.recv(min(data_args_size, buf_size))

    return data_fn, pickle.loads(data_args_bytes)