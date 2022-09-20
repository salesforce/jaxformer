# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from jaxformer.models.decoder.inter.model import create_model as create_model_decoder
from jaxformer.models.decoder.inter.checkpoint import try_save_ckpt as try_save_ckpt_decoder, load_ckpt as load_ckpt_decoder

def create_model(config):
    model_type = config['model_type']

    if model_type == 'decoder':
        return create_model_decoder(config), try_save_ckpt_decoder, load_ckpt_decoder
    else:
        raise Exception(model_type)
