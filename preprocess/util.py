# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import time
from multiprocessing.pool import ThreadPool

def par_map(f, args):
    return ThreadPool(processes=len(args)).map(f, args)


class print_time():
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        print(self.task)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.task} took {time.time()-self.t:.02f}s', flush=True)