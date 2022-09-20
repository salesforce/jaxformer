# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P

import haiku as hk

from .positional import fixed_pos_embedding, apply_rotary_pos_emb


# constants

ATTN_MASK_VALUE = -1e10

# helpers

LayerNorm = partial(hk.LayerNorm, create_scale=True, create_offset=True, axis=-1)



class FeedForward(hk.Module):
    def __init__(self, *, name, dim, init_stddev, ff_mult=4):
        super().__init__(name=name)

        self.proj_in = hk.Linear(dim * ff_mult, name='proj_in')
        self.proj_out = hk.Linear(dim, w_init=hk.initializers.TruncatedNormal(stddev=init_stddev / np.sqrt(dim)), name='proj_out')


    def __call__(self, x):

        x = self.proj_in(x)
        x = jax.nn.gelu(x)
        x = self.proj_out(x)

        return x


class Block(hk.Module):
    def __init__(self, *, name, dim, n_head, d_head, d_rotary, mp_num, init_scale=1.):
        super().__init__(name=name)

        self.ln = LayerNorm(name='ln')
        self.attn = CausalSelfAttention(name='attn', dim_out=dim, n_head=n_head, dim_head=d_head, d_rotary=d_rotary, init_stddev=init_scale, mp_num=mp_num)
        self.ff = FeedForward(name='ff', dim=dim, init_stddev=init_scale)

    def __call__(self, x):
        x = self.ln(x)

        attn_out = self.attn(x)
        ff_out = self.ff(x)

        assert x.shape == attn_out.shape == ff_out.shape

        x_out = attn_out + ff_out

        return x_out


class Projection(hk.Module):
    def __init__(self, n_vocab, name=None):
        super().__init__(name=name)

        self.n_vocab = n_vocab

        self.to_logits = hk.Sequential([
            LayerNorm(),
            hk.Linear(n_vocab)
        ])

    def __call__(self, x):
        return self.to_logits(x)


class CausalSelfAttention(hk.Module):
    def __init__(self, *, name, dim_out, n_head, dim_head, d_rotary, init_stddev, mp_num):
        super().__init__(name=name)

        self.n_head = n_head
        self.dim_head = dim_head
        self.mp_num = mp_num
        self.d_rotary = d_rotary

        self.proj_qkv = hk.Linear(n_head * dim_head * 3, with_bias=False, name='proj_qkv')
        self.proj_out = hk.Linear(dim_out, with_bias=False, w_init=hk.initializers.TruncatedNormal(stddev=init_stddev / jnp.sqrt(dim_out)), name='proj_out')


    def shard_heads(self, x):
        # TODO(enijkamp): rewrite
        reshaped = x.reshape(x.shape[:-1] + (self.n_head//self.mp_num, self.dim_head))

        # TODO(enijkamp): shard mp over logical heads
        reshaped = reshaped.reshape(x.shape[:-2] + (-1, ) + reshaped.shape[-1:])

        return reshaped


    def __call__(self, x):

        x_in_shape = x.shape

        n = x.shape[1]

        qkv = self.proj_qkv(x)
        qkv_split = jnp.reshape(qkv, qkv.shape[:-1] + (self.mp_num, -1))

        local_dim = self.dim_head * self.n_head // self.mp_num
        q, v, k = jnp.split(qkv_split, [local_dim, local_dim * 2], axis=-1)

        q = self.shard_heads(q)
        v = self.shard_heads(v)
        k = self.shard_heads(k)

        k_rot = k[:, :, :, :self.d_rotary]
        k_pass = k[:, :, :, self.d_rotary:]

        q_rot = q[:, :, :, :self.d_rotary]
        q_pass = q[:, :, :, self.d_rotary:]

        # TODO(enijkamp): rewrite resharding
        sincos = fixed_pos_embedding(k_rot, seq_dim=1)
        q_rot = apply_rotary_pos_emb(q_rot, sincos)
        k_rot = apply_rotary_pos_emb(k_rot, sincos)

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attn_scale = np.sqrt(self.dim_head).astype(k.dtype)
        attn_logits = jnp.einsum('b i h d, b j h d -> b h i j', q, k) / attn_scale

        causal_mask = jnp.triu(jnp.ones((n, n), dtype=bool), 1)
        attn_logits = jnp.where(causal_mask, ATTN_MASK_VALUE, attn_logits)

        # TODO(enijkamp): consider
        # attn_logits = attn_logits - jax.stop_gradient(jnp.amax(attn_logits, axis = -1, keepdims = True))

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_vec = jnp.einsum('b h i j, b j h d -> b i h d', attn_weights, v)

        # TODO(enijkamp): simplify
        attn_vec_sharded = attn_vec.reshape(attn_vec.shape[:2] + (self.mp_num, self.n_head//self.mp_num, -1))
        attn_vec = attn_vec.reshape(attn_vec_sharded.shape[:2] + (self.mp_num, -1))

        out = attn_vec.reshape(attn_vec.shape[:-2] + (-1,))

        x_out = self.proj_out(out)

        assert x_in_shape == x_out.shape, f'x_in_shape={x_in_shape} != x_out_shape={x_out.shape}'

        return x_out


# TODO(enijkamp): rewrite using hk.Embed()
class EmbeddingSharded(hk.Module):
    def __init__(self, in_dim, out_dim, name=None):
        super().__init__(name=name)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.proj = hk.Linear(self.out_dim, w_init=hk.initializers.TruncatedNormal(stddev=1 / np.sqrt(in_dim)), name='proj')

    def __call__(self, x):
        input_onehot = jax.nn.one_hot(x, self.in_dim)
        proj_out = self.proj(input_onehot)

        return proj_out