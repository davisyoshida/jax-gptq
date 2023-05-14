from collections import namedtuple
from contextlib import nullcontext
from functools import partial, wraps

import jax
import jax.numpy as jnp

class QuantizedMatrix:
    def __init__(self, int_weight, zero, scale, contraction_axis=0):
        self.int_weight = int_weight
        self.zero = zero
        self.scale = scale
        self.contraction_axis = contraction_axis

    @property
    def shape(self):
        result = quant_matrix_shape(self)
        return result

    @property
    def dtype(self):
        return self.zero.dtype

jax.tree_util.register_pytree_with_keys(
    QuantizedMatrix,
    lambda x: ([
        ('int_weight', x.int_weight),
        ('zero', x.zero),
        ('scale', x.scale),
    ], x.contraction_axis),
    lambda contraction_axis, xs: QuantizedMatrix(*xs, contraction_axis=contraction_axis),
)

def quant_matrix_shape(quantized_matrix, bits=4):
    mat = quantized_matrix.int_weight
    ele_width = mat.dtype.itemsize * 8

    contraction_dim = quantized_matrix.contraction_axis
    shape = list(mat.shape)
    shape[contraction_dim] *= ele_width // bits

    return tuple(shape)

def _cholesky_inv(mat):
    cho, upper = jax.scipy.linalg.cho_factor(mat)
    hinv = jax.scipy.linalg.cho_solve((cho, upper), jnp.eye(mat.shape[0]))
    hinv_cholesky = jnp.linalg.cholesky(hinv)
    return hinv_cholesky
    #return hinv

@partial(jax.jit, static_argnums=(1,))
def _calc_mean(xs, dtype):
    means = [jnp.mean(x.astype(dtype).reshape(-1, x.shape[-1]), axis=0) for x in xs]
    mean_of_means = jnp.mean(jnp.stack(means), axis=0)
    return mean_of_means

@partial(jax.jit, donate_argnums=(0,1,2))
def _accumulate_H_step(H, running_mean, n_samples, batch):
    if len(batch.shape) > 2:
        batch = batch.reshape(-1, batch.shape[-1])

    batch_size = batch.shape[0]
    new_n_samples = n_samples + batch_size

    batch = batch.astype(H.dtype)

    new_mean = running_mean + jnp.sum(batch - running_mean, axis=0) / new_n_samples

    x_val = batch - running_mean
    y_val = batch - new_mean

    H += x_val.T @ y_val

    return H, new_mean, new_n_samples

def accumulate_H(xs, use_fp64=False):
    """
    The method here differs from the original GPT-Q implementation.
    Instead of calculating the mean of XX^T directly, I instead calculate
    the variance using a running approximation of the mean.
    The second moment can then be recovered by adding the outer product
    of the mean vector with itself. The factor of 2 from the original
    paper doesn't matter since the algorithm is invariant to
    scaling the Hessian.
    """

    dtype = jnp.float64 if use_fp64 else jnp.float32
    #feature_mean = _calc_mean(xs, dtype)

    dim = xs[0].shape[-1]
    #N = sum(x.shape[0] for x in xs)
    H = jnp.zeros((dim, dim), dtype=dtype)
    n_samples = jnp.zeros((), dtype)

    running_mean = jnp.zeros(dim, dtype=dtype)

    for batch in xs:
        H, running_mean, n_samples = _accumulate_H_step(H, running_mean, n_samples, batch)

    H /= n_samples
    H += jnp.outer(running_mean, running_mean)

    return H

def get_quantize_params(weight, bits=4):
    params = {
        'maxq': 2**bits - 1,
    }
    xmin = jnp.minimum(0, jnp.min(weight))
    xmax = jnp.maximum(0, jnp.max(weight))
    xmin = jnp.where(xmin == 0, -1, xmin)
    xmax = jnp.where(xmax == 0, 1, xmax)
    params['scale'] = (xmax - xmin) / params['maxq']
    params['zero'] = jnp.round(-xmin / params['scale'])

    return params

get_col_quantize_params = jax.vmap(get_quantize_params, in_axes=1, out_axes={'scale': 0, 'zero': 0, 'maxq': None})

def quantize_value(zero, scale, maxq, value):
    q = jnp.clip(
        jnp.round(value / scale) + zero,
        0,
        maxq
    )
    return scale * (q - zero)

batch_quantize = jax.vmap(quantize_value, (0, 0, None, 0))

@partial(jax.jit, static_argnums=(4,))
def _process_block(W, Hinv, start, quantize_params, block_size):
    block = jax.lax.dynamic_slice_in_dim(W, start, block_size, axis=0)
    h_block = jax.lax.dynamic_slice(Hinv, (start, start), (block_size, block_size))
    quantized_rows = []
    errors = []
    for i in range(block_size):
        w = block[i, :]
        d = h_block[i, i]

        q = batch_quantize(quantize_params['zero'], quantize_params['scale'], quantize_params['maxq'], w)
        quantized_rows.append(q)

        error = (w - q) / d
        errors.append(error)

        update = (h_block[i:, i:i+1] @ error[None, :]).astype(block.dtype)
        block = block.at[i:, :].add(-update)

    q_block = jnp.stack(quantized_rows, axis=0)

    errors = jnp.stack(errors, axis=0)

    return q_block, errors

@partial(jax.jit, donate_argnums=(0, 1), static_argnums=2)
def _damp_and_invert(W, H, damping):
    dead = jnp.diag(H) == 0
    H = jnp.where(jnp.diag(dead), 1, H)
    W = jnp.where(dead[:, None], 0, W)

    mean_diag = jnp.mean(jnp.diag(H))
    positions = jnp.arange(H.shape[0])
    H = H.at[positions, positions].add(mean_diag * damping)

    Hinv = _cholesky_inv(H)
    any_nan = jnp.any(jnp.isnan(Hinv))
    #Hinv = jnp.linalg.inv(H)
    return W, Hinv, any_nan

@partial(jax.jit, donate_argnums=(0, 1))
def _permute(W, H):
    perm = jnp.argsort(jnp.diag(H))[::-1]
    W = W[perm, :]
    H = H[perm][:, perm]
    return W, H, perm

@partial(jax.jit, donate_argnums=(0,))
def _unpermute(Q, perm):
    invperm = jnp.argsort(perm)
    return Q[invperm, :]

def gptq(W, xs, block_size=128, actorder=False, damping=0.01, use_fp64=False):
    orig_type = W.dtype
    W = W.astype(jnp.float32)
    quantize_params = get_col_quantize_params(W)
    with (jax.experimental.enable_x64() if use_fp64 else nullcontext()):
        H = accumulate_H(xs, use_fp64=use_fp64)

        perm = None
        if actorder:
            W, H, perm = _permute(W, H)

        W, Hinv, any_nan = _damp_and_invert(W, H, damping)
        if any_nan:
            xs_nan = any(jnp.any(jnp.isnan(x)) for x in xs)
            raise ValueError('NaN when computing Hinv. Features NaN?: {xs_nan}')
        Hinv = Hinv.astype(jnp.float32)

    blocks = []
    carry = W
    for start in range(0, W.shape[0], block_size):
        end = start + block_size
        q_block, errors = _process_block(W, Hinv, start, quantize_params, block_size)
        update = (Hinv[end:, start:end] @ errors).astype(W.dtype)
        W = W.at[end:, :].add(-update)
        blocks.append(q_block)

    Q = jnp.concatenate(blocks, axis=0, dtype=orig_type)
    if perm is not None:
        Q = _unpermute(Q, perm)

    return Q, quantize_params

def pack_matrix(Q, params, contraction_axis=0):
    scale = params['scale']
    zero = params['zero']

    expanded_scale = jnp.expand_dims(scale, axis=contraction_axis)
    expanded_zero = jnp.expand_dims(zero, axis=contraction_axis)

    int_matrix = jnp.round(Q / expanded_scale + expanded_zero).astype(jnp.uint8)
    packed = pack_along_axis(contraction_axis, int_matrix,)
    return QuantizedMatrix(
        int_weight=packed,
        zero=zero,
        scale=scale,
        contraction_axis=contraction_axis
    )

def unpack_matrix(weight):
    expanded_zero = jnp.expand_dims(weight.zero, axis=weight.contraction_axis)
    expanded_scale = jnp.expand_dims(weight.scale, axis=weight.contraction_axis)

    unpacked = unpack_along_axis(weight.contraction_axis, weight.int_weight)
    return (unpacked - expanded_zero) * expanded_scale

def _pack(w_int):
    assert len(w_int.shape) == 1
    pack_dtype = jnp.uint8
    ele_width = pack_dtype.dtype.itemsize * 8
    bits = 4
    vals_per_int = ele_width // bits

    result = jnp.zeros(w_int.shape[0] // vals_per_int, dtype=pack_dtype)

    for offset in range(vals_per_int):
        result = result | (w_int[offset::vals_per_int] << (bits * offset)).astype(pack_dtype)

    return result

def _unpack(packed):
    assert len(packed.shape) == 1
    bits = 4
    ele_width = packed.dtype.itemsize * 8
    vals_per_int = ele_width // bits
    result = jnp.zeros(packed.shape[0] * vals_per_int, dtype=jnp.uint8)

    mask = (1 << bits) - 1
    for offset in range(vals_per_int):
        result = result.at[offset::vals_per_int].set(
            jnp.uint8(packed >> (bits * offset) & mask)
        )
    return result

def vmap_all_but_one(f, axis):
    @wraps(f)
    def inner(arg):
        n_dim = len(arg.shape)
        if axis >= n_dim:
            raise ValueError(f'Axis {axis} is out of bounds for array of dimension {n_dim}')

        fn = f
        vmap_dim = 1
        for i in reversed(range(n_dim)):
            if i == axis:
                vmap_dim = 0
            else:
                fn = jax.vmap(fn, vmap_dim, vmap_dim)
        return fn(arg)
    return inner

def pack_along_axis(axis, w):
    return vmap_all_but_one(_pack, axis)(w)

def unpack_along_axis(axis, w):
    return vmap_all_but_one(_unpack, axis)(w)

pack_rowwise = jax.vmap(_pack)
pack_colwise = jax.vmap(_pack, 1, out_axes=1)

unpack_rowwise = jax.vmap(_unpack)
unpack_colwise = jax.vmap(_unpack, 1, out_axes=1)
