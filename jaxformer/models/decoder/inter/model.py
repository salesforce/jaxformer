# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import thread_resources
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit, with_sharding_constraint

import optax
import haiku as hk

from jaxformer.utils import to_f32, to_bf16, global_norm, pjit_noop, with_sharding_constraint_noop, create_lr_schedule_gpt3_fn, add_decayed_weights_exclude_ln_and_bias

from .modules import EmbeddingSharded, Block, Projection


def create_model(config):

    ## optimizer

    lr_schedule_fn = create_lr_schedule_gpt3_fn(config['opt_warmup_steps'], config['opt_anneal_steps'], config['opt_lr_max'], config['opt_lr_end'])

    optimizer = optax.chain(
        optax.scale(1 / config['opt_gradient_accumulation_steps']),
        # TODO(enijkamp): make sure this is jnp.where() in https://github.com/deepmind/optax/blob/3d666d92211dc0e2689c3acf82656d6286952733/optax/_src/clipping.py#L100
        optax.clip_by_global_norm(config['opt_clip_by_global_norm']),
        optax.scale_by_adam(),
        add_decayed_weights_exclude_ln_and_bias(config['opt_weight_decay']),
        optax.scale(-1),
        optax.scale_by_schedule(lr_schedule_fn)
    )

    ## override pjit

    if config['debug_emulate_tpu_on_cpu']:
        maybe_pjit = pjit_noop
        maybe_with_sharding_constraint = with_sharding_constraint_noop
    else:
        maybe_pjit = pjit
        maybe_with_sharding_constraint = with_sharding_constraint


    ## model

    model = TransformerDecoder(config=config, maybe_pjit=maybe_pjit, maybe_with_sharding_constraint=maybe_with_sharding_constraint, optimizer=optimizer)

    return model, optimizer, lr_schedule_fn


def loss(vocab_size, logits, y, z_loss=1e-4):

    # TODO(enijkamp): in theory, this should improve numerical stability. However, (1) causes issues with the magnitude of gradient over time, (2) causes NaN on TPU-v3.
    # (1)
    # logits -= jax.lax.stop_gradient(logits.max(-1, keepdims=True))
    # (2)
    # logits -= logits.max(-1, keepdims=True)

    y_onehot = jax.nn.one_hot(y, vocab_size)
    logits_sum = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    log_softmax = logits - logits_sum
    loss = -jnp.sum(y_onehot * log_softmax, axis=-1)
    log_z = jnp.squeeze(logits_sum, axis=-1)
    loss += z_loss * jax.lax.square(log_z)

    return loss


def create_sharding_fns(config):

    ##  policy to partition tensors without 'dp'

    def shard_state_fn(shape_dtype, parallel):
        if shape_dtype.ndim == 0:
            return P()
        if shape_dtype.ndim == 1:
            return P(None)
        elif shape_dtype.shape == (config['model_vocab_size'], config['model_dim']):
            return P(parallel, None)
        elif shape_dtype.shape == (config['model_dim'], config['model_vocab_size']):
            return P(None, parallel)
        elif shape_dtype.shape[0] == config['model_layers']:
            if shape_dtype.ndim == 2:
                return P(None, None)
            elif shape_dtype.ndim == 3:
                matrix_size = shape_dtype.shape[1:]
                print(f'shard_strategy -> {shape_dtype}')
                if matrix_size[0] < matrix_size[1]:
                    return P(None, None, parallel)
                else:
                    return P(None, parallel, None)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()


    ##  policy is to replicate gradient along 'dp', but also take 'mp' into account

    def shard_grad_fn(shape_dtype, parallel):
        if shape_dtype.ndim == 0:
            return P('dp', None)
        if shape_dtype.ndim == 1:
            return P('dp', None)
        elif shape_dtype.shape == (config['model_vocab_size'], config['model_dim']):
            return P('dp', parallel, None)
        elif shape_dtype.shape == (config['model_dim'], config['model_vocab_size']):
            return P('dp', None, parallel)
        elif shape_dtype.shape[0] == config['model_layers']:
            if shape_dtype.ndim == 2:
                return P('dp', None, None)
            elif shape_dtype.ndim == 3:
                matrix_size = shape_dtype.shape[1:]
                print(f'shard_strategy -> {shape_dtype}')
                if matrix_size[0] < matrix_size[1]:
                    return P('dp', None, None, parallel)
                else:
                    return P('dp', None, parallel, None)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    return shard_state_fn, shard_grad_fn


def create_transformer_fns(config, dp, mp, with_sharding_constraint, optimizer=None):

    ## model

    def embedding(x):
        # x = with_sharding_constraint(x, P('pt', None))
        return EmbeddingSharded(in_dim=config['model_vocab_size'], out_dim=config['model_dim'])(x)

    def block():
        return Block(name=None, dim=config['model_dim'], n_head=config['model_heads'], d_head=config['model_head_dim'], d_rotary=config['model_pe_rotary_dims'], mp_num=mp, init_scale=2. / config['model_layers'])

    def residual(x):
        out = x + block()(x)
        # return with_sharding_constraint(out, P('pt', None, 'mp'))
        return out

    def transformer(x):
        return hk.remat(residual, prevent_cse=False)(x)

    def projection(x):
        return Projection(config['model_vocab_size'])(x)


    ## init

    def init(key, x):
        embed_init_fn = hk.transform(hk.experimental.optimize_rng_use(embedding)).init
        transformer_init_fn = hk.transform(hk.experimental.optimize_rng_use(transformer)).init
        projection_init_fn = hk.transform(hk.experimental.optimize_rng_use(projection)).init

        def init_scan_fn(key, x):
            new_key, key = jax.random.split(key)
            return new_key, transformer_init_fn(key, x)

        e_key, t_key, p_key = jax.random.split(key, 3)

        input_shape = (config['model_layers'],) + x.shape + (config['model_dim'],)

        model = {
            'embed': embed_init_fn(e_key, x),
            'transformer': jax.lax.scan(init_scan_fn, t_key, xs=jax.random.uniform(t_key, input_shape, dtype=jnp.float32))[1],
            'proj': projection_init_fn(p_key, jax.random.uniform(t_key, input_shape[1:], dtype=jnp.float32)),
        }

        output_state = {
            'model': model,
            'step': np.array(0),
        }

        if optimizer:
            output_state['opt'] = optimizer.init(to_f32(model))

        return output_state


    ## train

    def create_train(shard_state_mp, shard_grad):

        def train_apply_fn(model, x, y):

            def train_loss(x, y):
                logits = Projection(config['model_vocab_size'])(x)
                return loss(vocab_size=config['model_vocab_size'], logits=logits, y=y).mean()

            embed_apply_fn = hk.without_apply_rng(hk.transform(embedding)).apply
            transformer_apply_fn = hk.without_apply_rng(hk.transform(transformer)).apply
            projection_apply_fn = hk.without_apply_rng(hk.transform(train_loss)).apply

            x = embed_apply_fn(model['embed'], x)
            x = to_bf16(x)

            def apply_scan_fn(x, layer_state):
                return to_bf16(transformer_apply_fn(layer_state, x)), None

            x = jax.lax.scan(apply_scan_fn, x, xs=model['transformer'])[0]

            return projection_apply_fn(model['proj'], x, y)


        def train(state, x, y):
            is_single = (x.shape[0] == 1)
            params_bf16 = with_sharding_constraint(to_bf16(state['model']), shard_state_mp)
            value_and_grad_fn = jax.value_and_grad(train_apply_fn, has_aux=False, allow_int=True)

            def grad_sum(grad_old, x_y):
                x, y = x_y
                loss, grad = value_and_grad_fn(params_bf16, x, y)
                return jax.tree_map(jnp.add, grad_old, grad), loss

            if is_single:
                loss, grad = value_and_grad_fn(params_bf16, x[0], y[0])
            else:
                grad, loss = jax.lax.scan(grad_sum, jax.tree_map(lambda x: jnp.zeros_like(x).astype(jnp.bfloat16), params_bf16), (x, y))

            grad = jax.tree_map(lambda x: x.reshape([1, *x.shape]), grad)
            grad = jax.tree_map(lambda x: jax.numpy.repeat(x, dp, axis=0), grad)
            grad = with_sharding_constraint(grad, shard_grad)
            grad = jax.tree_map(lambda x: jax.numpy.mean(x, axis=0), grad)

            grad_global_norm = global_norm(grad)
            
            updates, new_opt_state = optimizer.update(grad, state['opt'], state['model'])

            return to_f32(loss), {
                'model': optax.apply_updates(state['model'], to_f32(updates)),
                'step': state['step'] + 1,
                'opt': new_opt_state,
            }, grad_global_norm

        return train


    return init, create_train


class TransformerDecoder:
    def __init__(self, config, maybe_pjit, maybe_with_sharding_constraint, optimizer=None):
        self.config = config

        ## mesh

        dp = thread_resources.env.shape['dp']
        mp = thread_resources.env.shape['mp']
        assert mp == config['tpu_cores']


        ## sharding

        def create_sharded_state(key, shard_state_fn, shard_grad_fn):        

            x = jax.random.uniform(next(key), (mp * dp, 16), minval=0, maxval=1).astype(jnp.uint32)

            state_shapes = jax.eval_shape(init, jax.random.PRNGKey(1), x)

            shard_state = {
                'step': P(),
                'model': jax.tree_map(partial(shard_state_fn, parallel='mp'), state_shapes['model']),
            }

            if 'opt' in state_shapes:
                shard_state['opt'] = jax.tree_map(partial(shard_state_fn, parallel='mp'), state_shapes['opt'])

            shard_grad = jax.tree_map(partial(shard_grad_fn, parallel='mp'), state_shapes['model'])
            shard_state_mp = jax.tree_map(partial(shard_state_fn, parallel=('mp')), state_shapes['model'])

            return shard_state, shard_grad, shard_state_mp


        ## model

        init, create_train = create_transformer_fns(config, dp, mp, maybe_with_sharding_constraint, optimizer)


        ## compile

        key = hk.PRNGSequence(1)

        shard_state_fn, shard_grad_fn = create_sharding_fns(config)
        shard_state, shard_grad, shard_state_mp = create_sharded_state(key, shard_state_fn, shard_grad_fn)

        train = create_train(shard_state_mp, shard_grad)

        self.init_pjit = maybe_pjit(init, in_axis_resources=(None, P('pt')), out_axis_resources=shard_state)
        self.train_pjit = maybe_pjit(train, in_axis_resources=(shard_state, P(None, 'pt'), P(None, 'pt')), out_axis_resources=(None, shard_state, None), donate_argnums=(0,))


        ## init

        x_shape = (max(dp // jax.process_count(), 1), config['model_seq_len'],)
        x = jax.random.uniform(next(key), x_shape, minval=0, maxval=config['model_vocab_size']).astype(jnp.uint32)

        self.state = self.init_pjit(next(key), x)


    def train(self, sample):
        loss, self.state, grad_global_norm = self.train_pjit(self.state, sample['x'], sample['y'])

        return self.state['step'], np.array(loss).mean(), np.array(grad_global_norm).mean().astype(np.float32)


    def profile(self, sample):

        jax.profiler.start_trace('profiles')
        loss = self.train_pjit(self.state, sample['x'], sample['y'])[0]
        loss.block_until_ready()
        jax.profiler.stop_trace()

        return self.state['step']


    def stats(self):
        params_num = hk.data_structures.tree_size(self.state['model'])
        params_size = hk.data_structures.tree_bytes(self.state['model'])
        print(f'Model params_num: {params_num}, params_size: {params_size / 1e6:.2f}MB')

        params_num_total = params_num
        params_size_total = params_size
        print(f'Model params_num_total: {params_num}, params_size_total: {params_size / 1e6:.2f}MB')

        return {'params_num': params_num, 'params_size': params_size, 'params_num_total': params_num_total, 'params_size_total': params_size_total}
