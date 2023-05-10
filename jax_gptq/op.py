from functools import partial

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from .kernel import matmul_4bit_quantized, matmul_4bit_quantized_traponse_b
from .gptq import unpack_matrix

def quantized_matmul(x, quantized_matrix, transpose_b=False):
    #unpacked = unpack_matrix(quantized_matrix)
    #return x @ (unpacked.T if transpose_b else unpacked)
    return _quantized_matmul(x, quantized_matrix, transpose_b)

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _quantized_matmul(x, quantized_matrix, transpose_b):
    leading_dims = None
    if len(x.shape) > 2:
        leading_dims = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
    zero = quantized_matrix.zero
    scale = quantized_matrix.scale
    w = quantized_matrix.int_weight
    M, K = x.shape

    if transpose_b:
        N, K2 = w.shape
        out_N = 2 * N
        stride_bn = K2
        stride_bk = 1
    else:
        K2, N = w.shape
        out_N = N
        K2 *= 2
        stride_bn = 1
        stride_bk = N
    assert K == K2, f'Inner dim mismatch: {K} != {K2}'

    out_struct = jax.ShapeDtypeStruct((M, out_N), jnp.float16)
    kernel = matmul_4bit_quantized_traponse_b if transpose_b else matmul_4bit_quantized

    result = jt.triton_call(
        x,
        w,
        zero,
        scale,
        out_shape=out_struct,
        kernel=kernel,
        grid=lambda META: (triton.cdiv(META['M'], META['BLOCK_SIZE_M']) * triton.cdiv(META['N'], META['BLOCK_SIZE_N']),),
        stride_am=K,
        stride_ak=1,
        stride_bn=stride_bn,
        stride_bk=stride_bk,
        stride_cm=out_N,
        stride_cn=1,
        M=M,
        N=N,
        K=K,
    )
    if leading_dims is not None:
        result = result.reshape(leading_dims + (result.shape[-1],))
    if not result.dtype == x.dtype:
        # TODO: Make kernel directly output correct dtype instead
        result = result.astype(x.dtype)
    return result

def qmatmul_fwd(x, quantized_matrix, transpose_b):
    #assert not transpose_b
    #return x @ unpack_matrix(quantized_matrix), quantized_matrix
    return _quantized_matmul(x, quantized_matrix, transpose_b), quantized_matrix

def qmatmul_bwd(transposed, quantized_matrix, y_bar):
    unpacked = unpack_matrix(quantized_matrix)
    return y_bar @ unpacked.T, None
    #return _quantized_matmul(y_bar, quantized_matrix, transpose_b=not transposed), None

_quantized_matmul.defvjp(qmatmul_fwd, qmatmul_bwd)
