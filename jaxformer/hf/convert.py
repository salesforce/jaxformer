import argparse
import io
import warnings
import os
import re
import json
from typing import Iterable, List, Union

from pathy import FluidPath, Pathy
import numpy as np
import torch

from smart_open import open

import jax
import jax.numpy as jnp
from jax.experimental import maps

from jaxformer.utils import print_time, pjit_noop, with_sharding_constraint_noop
from jaxformer.models.decoder.inter.model import TransformerDecoder
from jaxformer.models.decoder.inter.checkpoint import load_ckpt


def process_args(
    input_ckpt: Union[FluidPath, str],
    config: Union[FluidPath, str],
    output_path: Union[FluidPath, str],
    dtype: str = "fp16",
    **kwargs,
):
    input_ckpt = Pathy.fluid(str(input_ckpt))
    assert input_ckpt.is_dir(), f'no such directory "{input_ckpt}"'
    config = Pathy.fluid(str(config))
    assert config.is_file(), f'no such file "{config}"'

    output_path = Pathy.fluid(str(output_path))
    output_path.mkdir(exist_ok=True)

    assert dtype in {"fp16", "fp32", "bf16"}
    np_dtype = np.float16
    torch_dtype = torch.float16
    if dtype != "fp16":
        warnings.warn(
            "WARNING: Dtype support other than fp16 is Experimental. Make sure to check weights after conversion to make sure dtype information is retained."
        )
        if dtype == "bf16":
            np_dtype = np.float32
            torch_dtype = torch.bfloat16
        elif dtype == "fp32":
            np_dtype = np.float32
            torch_dtype = torch.float32

    return input_ckpt, config, output_path, np_dtype, torch_dtype


def tree_flatten_with_names(pytree, is_leaf, path="", to_id=id):
    id_to_name = {}
    if getattr(pytree, "items", None):
        for k, v in pytree.items():
            k_path = f"{path}/{k}"
            if is_leaf(v):
                id_to_name[to_id(v)] = k_path
            else:
                id_to_name = {
                    **id_to_name,
                    **tree_flatten_with_names(v, is_leaf=is_leaf, path=k_path),
                }
    elif getattr(pytree, "__getitem__", None):
        for v in pytree:
            if is_leaf(v):
                id_to_name[to_id(v)] = path
            else:
                id_to_name = {
                    **id_to_name,
                    **tree_flatten_with_names(v, is_leaf=is_leaf, path=path),
                }
    else:
        id_to_name[to_id(pytree)] = path
    return id_to_name


def tree_leaves_with_names(pytree, to_id=id):
    leaves = jax.tree_leaves(pytree)
    is_leaf = lambda x: not isinstance(x, list) and to_id(x) in [
        to_id(x) for x in leaves
    ]
    return tree_flatten_with_names(pytree, is_leaf)


def get_tree_leaves_names_reduced(pytree) -> List[str]:

    leaves_ids = tree_leaves_with_names(pytree, to_id=id)
    leaves = jax.tree_leaves(pytree)
    return [leaves_ids[id(l)] for l in leaves]


layer_2_hf_inner_module_id = {
    "attn.proj_qkv": "attn.qkv_proj",
    "attn.proj_out": "attn.out_proj",
    "ff.proj_in": "mlp.fc_in",
    "ff.proj_out": "mlp.fc_out",
    "block.ln": "ln_1",
}

projection_layer_2_hf_id_start = {
    "linear": "lm_head",
    "layer_norm": "transformer.ln_f",
}


def leave_name_to_hf_layer_id(leaf_name: str):

    match = re.search(
        r"\/(.*)\/(?P<module_name>.*)\/~\/(?P<layer_name>.*)\/(?P<wb>.*)",
        leaf_name,
    )

    assert match, f'couldn\'t match pattern against: "{leaf_name}"'

    module_name = match["module_name"]
    layer_name = match["layer_name"]
    wb = match["wb"]

    print(f'{leaf_name} -> {module_name}/{layer_name}/{wb}')

    if wb in {"w", "scale"}:
        weight_or_bias = "weight"
    elif wb in {"b", "offset"}:
        weight_or_bias = "bias"
    else:
        raise NotImplementedError(
            f"unknown weight/bais type identifier \"{wb}\" at end of: '{leaf_name}'"
        )



    if module_name == "embedding_sharded":
        return f"transformer.wte.{weight_or_bias}"



    elif module_name == "projection":
        return f"{projection_layer_2_hf_id_start[layer_name]}.{weight_or_bias}"



    elif module_name == "attn":
        hf_inner_module_id = layer_2_hf_inner_module_id[f'{module_name}.{layer_name}']
        return "transformer.h.{}" + f".{hf_inner_module_id}.{weight_or_bias}"

    elif module_name == "ff":
        hf_inner_module_id = layer_2_hf_inner_module_id[f'{module_name}.{layer_name}']
        return "transformer.h.{}" + f".{hf_inner_module_id}.{weight_or_bias}"

    elif module_name == "block":
        hf_inner_module_id = layer_2_hf_inner_module_id[f'{module_name}.{layer_name}']
        return "transformer.h.{}" + f".{hf_inner_module_id}.{weight_or_bias}"



    else:
        raise NotImplementedError(
            f"unknown leaf module type \"{module_name}\" in: '{leaf_name}'"
        )


def read_npz(fpath: FluidPath):
    with fpath.open("rb") as f:
        buf = f.read()
        f_io = io.BytesIO(buf)
        deserialized = np.load(
            f_io,
        )
        assert isinstance(
            deserialized, np.lib.npyio.NpzFile
        ), f"Not an npz file {type(deserialized)=} {f=}"

        arrays = []
        for i in deserialized:
            arr = deserialized[i]
            assert isinstance(arr, np.ndarray), f"Not a np.ndarray {type(arr)=} {f=}"
            arrays.append(arr)
        return arrays


def save_pytree_as_hf(
    pytree,
    input_ckpt: FluidPath,
    shards_in: int,
    output_path: FluidPath,
    n_layers: int = 28,
    np_dtype: type = np.float16,
    torch_dtype: torch.dtype = torch.float16,
    n_seq: int = 2048,
):
    old_leave_shapes = [old.shape for old in jax.tree_flatten(pytree)[0]]
    leave_names = get_tree_leaves_names_reduced(pytree)
    print(leave_names)
    del pytree

    assert len(old_leave_shapes) == len(
        leave_names
    ), f"{len(old_leave_shapes)=}  {len(leave_names)=}"

    # TODO(enijkamp): hack since xmap() on pjit() emulation replicates weights in all shards, load first copy
    print('loading weights')
    loaded_shards_in = read_npz(input_ckpt / 'model' / '0.npz')
    print('loaded weights')

    assert len(leave_names) == len(old_leave_shapes)
    assert len(leave_names) == len(loaded_shards_in)

    hf_checkpoint = {}
    wte_first = None

    for i in range(len(leave_names)):

        print('#' * 20)
        print(i, len(leave_names))

        x = loaded_shards_in[i]
        leave_name = leave_names[i]
        old_shape = old_leave_shapes[i]

        print(leave_name)
        print(old_shape)
        print(x.shape)

        hf_layer_id = leave_name_to_hf_layer_id(leave_name)
        print(hf_layer_id)

        assert old_shape == x.shape

        if hf_layer_id.startswith("transformer.h"):
            print('unfold layers')

            assert x.shape[0] == n_layers

            if 'ln_1' in hf_layer_id:
                print(hf_layer_id)
                print(x[0])

            for layer_i in range(n_layers):
                x_layer = x[layer_i]

                # TODO(enijkamp): why?
                # x = torch.tensor(x.squeeze(0), dtype=torch_dtype).T
                x_layer = torch.tensor(x_layer, dtype=torch_dtype).T

                hf_checkpoint[hf_layer_id.format(layer_i)] = x_layer
        
        else:

            # TODO(enijkamp): why?
            # x = torch.tensor(x.squeeze(0), dtype=torch_dtype).T
            x = torch.tensor(x, dtype=torch_dtype).T

            if hf_layer_id.startswith("transformer.wte"):
                x = x.T
                if wte_first is None:
                    wte_first = x
                    continue
                else:
                    x = x + wte_first
                    hf_layer_id = "transformer.wte.weight"

            hf_checkpoint[hf_layer_id] = x

    attn_bias_weights = torch.tril(torch.ones((n_seq, n_seq), dtype=torch.bool)).view(
        1, 1, n_seq, n_seq
    )
    attn_masked_bias_weights = torch.tensor(-1e9, dtype=torch_dtype)

    for i in range(n_layers):
        # TODO(enijkamp): fix for HF format
        # hf_checkpoint[f"transformer.h.{i}.attn.bias"] = attn_bias_weights
        hf_checkpoint[f"transformer.h.{i}.attn.causal_mask"] = attn_bias_weights
        hf_checkpoint[f"transformer.h.{i}.attn.masked_bias"] = attn_masked_bias_weights

    with open(output_path / "pytorch_model.bin", mode="wb") as f:
        torch.save(hf_checkpoint, f)


def save_config_to_hf_format(params: dict, torch_dtype: str, output_path: FluidPath):

    config = {
        "activation_function": "gelu_new",
        "architectures": ["CodeGenForCausalLM"],
        "attn_pdrop": 0.0,
        "bos_token_id": 1,
        "embd_pdrop": 0.0,
        "eos_token_id": 2,
        "gradient_checkpointing": False,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "codegen",
        "n_embd": params["model_dim"],
        "n_head": params["model_heads"],
        "n_layer": params["model_layers"],
        "n_positions": params["model_seq_len"],
        "rotary_dim": params["model_pe_rotary_dims"],
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "transformers_version": "4.10.0.dev0",
        "tokenizer_class": "GPT2Tokenizer",
        "task_specific_params": {
            "text-generation": {"do_sample": True, "temperature": 1.0, "max_length": 50}
        },
        "torch_dtype": str(torch_dtype).split(".")[-1],
        "use_cache": True,
        "vocab_size": params["model_vocab_size"],
    }

    with open(output_path / "config.json", mode="w") as f:
        json.dump(config, f, indent=2)


def save_sharded_to_hf_format(
    input_ckpt: Union[FluidPath, str],
    config: dict,
    output_path: Union[FluidPath, str],
    np_dtype,
    torch_dtype
):

    devices = np.array([jax.devices()[0]]).reshape((1, 1, 1))
    with maps.Mesh(devices, ("dp", "pt", "mp")):
        config_local = config.copy()
        config_local["tpu_cores"] = maps.thread_resources.env.shape["mp"]

        model = TransformerDecoder(config=config_local, maybe_pjit=pjit_noop, maybe_with_sharding_constraint=with_sharding_constraint_noop, optimizer=None)

        save_pytree_as_hf(
            model.state["model"],
            input_ckpt=input_ckpt,
            shards_in=config["tpu_cores"],
            output_path=output_path,
            n_layers=config["model_layers"],
            np_dtype=np_dtype,
            torch_dtype=torch_dtype,
            n_seq=config["model_seq_len"],
        )


##############################################################################################
# main


def setup_jax(cpu=True):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    if cpu:
        jax.config.update("jax_platform_name", "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/codegen_1B_mono.json')
    parser.add_argument('--step', type=int, default=150000)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    args.input_ckpt = f'{config["ckpt_dir"]}/{args.step}'
    args.output_path = f'{config["ckpt_dir"]}_torch/{args.step}'

    return args


if __name__ == '__main__':

    with print_time('preamble'):
        setup_jax(cpu=True)

        args = parse_args()

        print('-' * 40)
        print(f'step={args.step}')
        print('-' * 40)

    with print_time('converting paramaters'):
        input_ckpt, config, output_path, np_dtype, torch_dtype = process_args(**vars(args))
        with open(config) as f:
            params = json.load(f)

    with print_time('serializing paramaters'):
        save_sharded_to_hf_format(input_ckpt, params, output_path, np_dtype, torch_dtype)
        save_config_to_hf_format(params, torch_dtype, output_path)

    print('done')
