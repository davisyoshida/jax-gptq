from itertools import product

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest

from jax_gptq import use_quantized
from jax_gptq.quantize_interpreter import quantize
from jax_gptq.gptq import pack_colwise, unpack_colwise, pack_matrix, unpack_matrix, get_col_quantize_params, gptq, QuantizedMatrix

@pytest.fixture
def simple_model():
    def f(w, x):
        return x @ w
    w = jax.random.normal(jax.random.PRNGKey(0), (256, 1024))
    orig_w = np.asarray(w)
    xs = [jax.random.normal(jax.random.PRNGKey(1), (32, 256), dtype=jnp.float16) for _ in range(4)]

    quantized = jax.device_put(quantize(f, w, xs, block_size=16), jax.devices('gpu')[0])
    return f, w, quantized, xs

def test_get_quantize_params():
    w = jnp.asarray([
        [0.1, 0.2, 0.3, 0.4],
        [-0.1, -0.1, -0.1, -0.1],
    ], dtype=jnp.float32)
    q_params = get_col_quantize_params(w)
    scale = q_params['scale']
    zero = q_params['zero']

    assert zero.shape == scale.shape == (4,)
    assert np.allclose(scale, (w[0, :] - w[1, :]) / 15)

def test_single():
    def f(w, x):
        return x @ w

    w = jax.random.normal(jax.random.PRNGKey(0), (256, 64))
    orig_w = np.asarray(w)

    xs = [jax.random.normal(jax.random.PRNGKey(1), (32, 256)) for _ in range(4)]

    fn = jax.vmap(f, (None, 0))

    orig_output = fn(w, xs[0])

    quantized = quantize(fn, w, xs, block_size=16)
    de_quantized = unpack_matrix(quantized)

    new_output = fn(de_quantized, xs[0])

    print(f'Scale: {quantized.scale} Zero: {quantized.zero}')

    diff = jnp.max(jnp.abs(orig_w - de_quantized))
    print(f'Max quantization err: {diff}')
    abs_error = jnp.abs(orig_output - new_output)
    relative_error = jnp.abs(abs_error / orig_output)
    print(f'Max relative error: {jnp.max(relative_error)}')
    print(f'Max absolute error: {jnp.max(jnp.abs(orig_output - new_output))}')
    print(f'Max error relative to max: {jnp.max(relative_error) / jnp.mean(jnp.abs(orig_output))}')
    print(f'Max error relative to max: {jnp.max(relative_error) / jnp.max(jnp.abs(orig_output))}')

def test_hk():
    hk.mixed_precision.set_policy(
        hk.Linear,
        jmp.Policy(
            compute_dtype=jnp.float16,
            output_dtype=jnp.float16,
            param_dtype=jnp.float16
        )
    )
    def f(x):
        for _ in range(3):
            x = hk.Linear(1024, with_bias=False)(x)
        return x

    in_dim = 256
    model = hk.without_apply_rng(hk.transform(f))
    params = model.init(jax.random.PRNGKey(0), jnp.ones(in_dim))

    xs = [jax.random.normal(jax.random.PRNGKey(i), (32, in_dim)) for i in range(64)]

    fn = jax.vmap(model.apply, (None, 0))

    orig_output = fn(params, xs[0])
    quantized_params = quantize(fn, params, xs)

    manual_result = xs[0]
    for i in range(3):
        layer_params = quantized_params[f'linear_{i}' if i > 0 else 'linear']
        manual_result = manual_result @ unpack_matrix(layer_params['w'])
    manual_result = jax.device_put(manual_result, jax.devices('gpu')[0])

    gpu_args = jax.device_put((quantized_params, xs[0]), jax.devices('gpu')[0])
    new_output = use_quantized(fn)(*gpu_args)
    #new_output = use_quantized(fn)(quantized_params, xs[0])

    abs_error = jnp.abs(orig_output - new_output)
    relative_error = jnp.abs(abs_error / orig_output)
    print(f'Max absolute error from manual calculation: {jnp.max(jnp.abs(manual_result - new_output))}')
    print(f'Max relative error: {jnp.max(relative_error)}')
    print(f'Max absolute error: {jnp.max(jnp.abs(orig_output - new_output))}')

    assert np.allclose(manual_result, new_output, atol=3e-3, rtol=0)

def test_transform():
    def f(w, x):
        return x @ w

    w = jax.random.normal(jax.random.PRNGKey(0), (2048, 2048))
    xs = [jax.random.normal(jax.random.PRNGKey(i), (32, 2048), dtype=jnp.float32) for i in range(64)]

    fn = jax.vmap(f, (None, 0))
    orig_result = jax.device_put(fn(w, xs[0]), jax.devices('gpu')[0])

    quantized_params = quantize(fn, w, xs, block_size=16)
    unpacked_matrix = unpack_matrix(quantized_params)

    gpu_args = jax.device_put((quantized_params, xs[0]), jax.devices('gpu')[0])

    manual_result = jax.device_put(xs[0] @ unpacked_matrix, jax.devices('gpu')[0])

    transform_result = use_quantized(fn)(*gpu_args)
    print(f'Gap to manual calculation: {np.max(np.abs(manual_result - transform_result))}')
    print(f'Gap to original: {np.max(np.abs(orig_result - transform_result))}')
    assert np.allclose(manual_result, transform_result, atol=1e-4, rtol=0)

def test_pack():
    w = jax.random.randint(jax.random.PRNGKey(0), (256, 64), 0, 16)
    packed = pack_colwise(w)
    print(packed.shape, packed.dtype)
    unpacked = unpack_colwise(packed)
    assert jnp.all(unpacked == w)

def test_pack_matrix():
    w = jax.random.normal(
        jax.random.PRNGKey(0),
        (256, 64),
    )
    xs = [jax.random.randint(jax.random.PRNGKey(i), (32, 256), 0, 16) for i in range(4)]
    quantized, qparams = gptq(w, xs, block_size=4)
    packed_w = pack_matrix(quantized, qparams)
    unpacked_w = unpack_matrix(packed_w)
    assert jnp.all(unpacked_w == quantized)

def test_remat(simple_model):
    f, _, w_q, xs = simple_model
    x = xs[0]

    expected = x @ unpack_matrix(w_q)

    fn = jax.jit(use_quantized(jax.checkpoint(f)))
    result = fn(w_q, x)
    diff = jnp.max(jnp.abs(result - expected))
    print(f'Max error: {diff}')
    assert np.allclose(result, expected, atol=0.03, rtol=0)

def test_grad(simple_model):
    _, _, w_q, xs = simple_model
    x = xs[0]

    def f(x, w):
        return jnp.sum((x @ w)[3])

    unpacked = unpack_matrix(w_q)

    grad_fn = jax.jit(jax.grad(use_quantized(jax.checkpoint(f))))
    grad = grad_fn(x, w_q)

    expected_grad = jax.grad(f)(x, unpacked)
    print(f'Grad: {grad}')
    print(f'Expected grad: {expected_grad}')
    print(f'Max error: {jnp.max(jnp.abs(grad - expected_grad))}')
    assert np.allclose(grad, expected_grad, atol=1e-1, rtol=0)

@pytest.mark.parametrize(
    ['w_contract', 'x_contract'],
    list(product(range(3), range(2)))
)
def test_weird_dots(w_contract, x_contract):
    w_shape = [47, 768]
    x_shape = [32]

    contract_dim_size = 256
    w_shape.insert(w_contract, contract_dim_size)
    x_shape.insert(x_contract, contract_dim_size)

    w = jax.random.normal(jax.random.PRNGKey(9999), w_shape)
    xs = [jax.random.normal(jax.random.PRNGKey(10 * i), x_shape) for i in range(10)]

    def f(w, x):
        return jax.lax.dot_general(
            w, x,
            (((w_contract,), (x_contract,)), ((), ())),
        )

    expected_output = f(w, xs[0])

    quantized_params = quantize(f, w, xs, block_size=16)
    quantized_output = use_quantized(f)(quantized_params, xs[0])

    print(f'Error: {jnp.max(jnp.abs(expected_output - quantized_output)):.3e}')
    mean_error = jnp.mean(jnp.abs(expected_output - quantized_output))
    print(f'Mean error: {mean_error:.3e}')
    assert (mean_error / jnp.mean(jnp.abs(expected_output))) < 0.1

def test_conv():
    in_channels = 32
    out_channels = 79
    batch_size = 11
    spatial_size = 13

    w = jax.random.normal(jax.random.PRNGKey(0), (
        out_channels,
        in_channels,
    ))

    xs = [jax.random.normal(jax.random.PRNGKey(i), (batch_size, in_channels, spatial_size)) for i in range(1, 20)]

    def matmul_f(w, x):
        return jax.lax.dot_general(w, x, (((1,), (1,)), ((), ())))

    def conv_f(w, x):
        return jax.lax.conv(x, w, (1,), 'VALID')


    quantized_params = quantize(matmul_f, w, xs, block_size=16)
    quantized_output = use_quantized(matmul_f)(quantized_params, xs[0])

    conv_quantized = quantize(conv_f, w[:, :, None], xs, block_size=16)

    unpacked_w = unpack_matrix(quantized_params)
    unpacked_kernel = unpack_matrix(conv_quantized)
    num_diff = jnp.sum(unpacked_w != unpacked_kernel[:, :, 0])
    assert num_diff < 10 # Can have some differences from non-determinism in quantization

    quanitzed_conv_output = use_quantized(conv_f)(conv_quantized, xs[0]).transpose(1, 0, 2)

    max_diff = jnp.max(jnp.abs(quantized_output - quanitzed_conv_output))

    print(f'Conv vs Matmul quantization diff: {max_diff:.3e}')
    assert max_diff < 1e-5

def test_transpose():
    w = jax.random.normal(jax.random.PRNGKey(9999), (256, 1024))
    w_copy = np.asarray(w)
    xs = [jax.random.normal(jax.random.PRNGKey(i), (256, 32)) for i in range(10)]

    def f(w, x):
        return w.T @ x

    orig_output = f(w, xs[0])

    quantized_params = quantize(f, w, xs, block_size=16)
    assert isinstance(quantized_params, QuantizedMatrix)

    quantized_output = use_quantized(f)(quantized_params, xs[0])

    unpacked = unpack_matrix(quantized_params)

    manual_result = unpacked.T @ xs[0]

    param_diff = jnp.mean(jnp.abs(unpack_matrix(quantized_params) - w_copy))
    assert param_diff < 0.15

    assert np.allclose(manual_result, quantized_output, atol=1e-4)

def test_reuse(simple_model):
    _, w, _, xs = simple_model
    def f(w, x):
        intermediate = w.T
        answer = w.T @ x.T
        answer += jnp.sum(w)
        return answer

    quantized_params = quantize(f, w, xs, block_size=16)
    assert isinstance(quantized_params, QuantizedMatrix)

    quantized_output = use_quantized(f)(quantized_params, xs[0])
