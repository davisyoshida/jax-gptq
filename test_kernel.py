import jax
import jax.numpy as jnp
import pytest

from jax_gptq import gptq
from jax_gptq.op import quantized_matmul

@pytest.mark.parametrize(
    'M,N,K,transpose',
    [(137, 909, 256, t) for t in (True, False)]#, (32, 1024, 2048), (37, 32000, 5120)]
)
def test_kernel(M, N, K, transpose):
    xs = [jax.random.normal(jax.random.PRNGKey(key), (M, K)) for key in range(1)]
    W = jax.random.normal(jax.random.PRNGKey(4), (K, N))

    quant_w, params = gptq.gptq(W, xs)
    packed = gptq.pack_matrix(quant_w, params)

    test_input = jax.random.normal(jax.random.PRNGKey(5), (M, N)) if transpose else xs[0]
    expected = jnp.matmul(test_input, (quant_w.T if transpose else quant_w))

    @jax.jit
    def f(x, packed):
        return quantized_matmul(x, packed, transpose_b=transpose)

    #jit_fn = quantized_matmul
    for _ in range(1):
        ret = f(test_input, packed)
        print(jnp.sum(ret))

    max_gap = jnp.max(jnp.abs(ret - expected))
    print(f'Result:\n{ret[15:19, -3:]}')
    print(f'Expected:\n{expected[15:19, -3:]}')
    print(f'Max gap: {max_gap}')
    assert jnp.allclose(ret, expected, atol=4e-2, rtol=0)
