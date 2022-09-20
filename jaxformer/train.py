# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

import json
import argparse

from .run.trainer import train

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/debug_cpu.json')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train(config=config)
    
    print('done.')