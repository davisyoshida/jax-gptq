import jax
import jax.numpy as jnp

from .gptq import QuantizedMatrix, quant_matrix_shape

def _shape_from_param(x):
    if isinstance(x, QuantizedMatrix):
        return jax.ShapedArray(quant_matrix_shape(x), jnp.float32, weak_type=True)
    if isinstance(x, jax.core.Trace):
        return jax.core.get_aval(x)
    return x

def quantized_params_to_shaped_arrays(params):
    """Map a pytree containing QuantizedMatrix instances to one in which those are replaced by
    ShapedArray instances of the correct shape
    """
    return jax.tree_map(
        _shape_from_param,
        params,
        is_leaf=lambda x: isinstance(x, QuantizedMatrix)
    )
