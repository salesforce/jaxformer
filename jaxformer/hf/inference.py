# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

'''
cd ~/jaxformer

python3.8 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip setuptools

pip3 install torch==1.9.0+cu111 transformers==4.21.3 accelerate==4.3.0 --find-links https://download.pytorch.org/whl/torch_stable.html

'''

import os
import re
import time
import random
import argparse
import json

import torch

from transformers import GPT2TokenizerFast, AutoConfig, AutoModelForSeq2SeqLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed

from jaxformer.hf.codegen.modeling_codegen import CodeGenForCausalLM


########################################################################
# util

class print_time:
    def __init__(self, desc):
        self.desc = desc

    def __enter__(self):
        print(self.desc)
        self.t = time.time()

    def __exit__(self, type, value, traceback):
        print(f'{self.desc} took {time.time()-self.t:.02f}s')


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def cast(model, fp16=True):
    if fp16:
        model.half()
    return model


def create_args(args=argparse.Namespace(), ds_config_path="config/ds_config_inf.json"):
    args.seed = 42

    with open(ds_config_path) as f:
        config = json.load(f)
    args.ds_config = config

    args.opt_steps_train = 1000
    args.model_seq_len = 2048
    args.model_vocab_size = 51200
    args.debug_fixed_batch = False
    args.debug_mock_data = False
    return args

# constants
MODELS_NL = ['codegen-350M-nl', 'codegen-2B-nl', 'codegen-6B-nl', 'codegen-16B-nl']
MODELS_PL = ['codegen-350M-multi', 'codegen-1B-multi', 'codegen-2B-multi', 'codegen-6B-multi', 'codegen-16B-multi', 'codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'codegen-16B-mono']
MODELS = MODELS_NL + MODELS_PL

########################################################################
# model

def create_model(ckpt):
    return CodeGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


########################################################################
# sample

def sample(
    device,
    model,
    tokenizer,
    context,
    pad_token_id,
    num_return_sequences=1,
    temp=0.2,
    top_p=0.95,
    max_length_sample=128,
    max_length=2048
):
    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    input_ids_len = input_ids.shape[1]
    assert input_ids_len < max_length

    with torch.no_grad():
        input_ids = input_ids.to(device)
        tokens = model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temp,
            max_length=input_ids_len + max_length_sample,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True,
        )
        text = tokenizer.batch_decode(tokens[:, input_ids_len:, ...])

    return text


def truncate(completion):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [
        re.compile(r, re.MULTILINE)
        for r in
        [
            '^#',
            re.escape('<|endoftext|>'),
            "^'''",
            '^"""',
            '\n\n\n'
        ]
    ]

    prints = list(re.finditer('^print', completion, re.MULTILINE))
    if len(prints) > 1:
        completion = completion[:prints[1].start()]

    defs = list(re.finditer('^def', completion, re.MULTILINE))
    if len(defs) > 1:
        completion = completion[:defs[1].start()]

    start_pos = 0

    terminals_pos = [pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1]
    if len(terminals_pos) > 0:
        return completion[:min(terminals_pos)]
    else:
        return completion


def test_truncate():
    assert truncate('\nif len_a > len_b:\n    result = a\nelse:\n    result = b\n\n\n\n#') == '\nif len_a > len_b:\n    result = a\nelse:\n    result = b'


########################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=MODELS, default='codegen-6B-multi')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', type=bool, default=True)
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--t', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--pad', type=int, default=50256)
    parser.add_argument('--context', type=str, default='def helloworld():')
    parser.add_argument('--ds-config-path', type=str, default='config/ds_config_inf.json')
    parser.add_argument('--ckpt-path', type=str, default='jaxformer/hf/checkpoints/')
    args = parser.parse_known_args()[0]

    args = create_args(args, args.ds_config_path)

    # preamble
    set_env()
    set_seed(args.rng_seed, deterministic=args.rng_deterministic)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    device = torch.device(args.device)
    ckpt = os.path.join(args.ckpt_path, args.model)

    config = AutoConfig.from_pretrained(ckpt)
    model_hidden_size = config.n_ctx
    train_batch_size = 1 * world_size

    args.ds_config["reduce_bucket_size"] = model_hidden_size * model_hidden_size
    args.ds_config["stage3_prefetch_bucket_size"] = 0.9 * model_hidden_size * model_hidden_size
    args.ds_config["stage3_param_persistence_threshold"] = 10 * model_hidden_size
    args.ds_config["train_batch_size"] = train_batch_size

    dschf = HfDeepSpeedConfig(args.ds_config)

    # load
    with print_time('loading parameters'):
        model = create_model(ckpt=ckpt)

    with print_time('loading deepspeed module'):
        ds_engine = deepspeed.initialize(model=model, config_params=args.ds_config)[0]
        ds_engine.module.eval()

    with print_time('loading tokenizer'):
        if args.model in MODELS_PL:
            tokenizer = create_custom_gpt2_tokenizer()
        else:
            tokenizer = create_tokenizer()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = args.pad

    # sample
    with print_time('sampling'):
        completion = sample(device=device,
                            model=model,
                            tokenizer=tokenizer,
                            context=args.context,
                            pad_token_id=args.pad,
                            num_return_sequences=args.batch_size,
                            temp=args.t,
                            top_p=args.p,
                            max_length_sample=args.max_length)[0]

        truncation = truncate(completion)

        print('=' * 100)
        print(completion)
        print('=' * 100)
        print(args.context+truncation)
        print('=' * 100)


if __name__ == '__main__':
    test_truncate()
    main()
    print('done.')
