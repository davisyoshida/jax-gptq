from functools import partial, wraps
import warnings

import jax
from jax.core import Literal
import jax.numpy as jnp
from jax.util import safe_map
import jaxlib

from .gptq import QuantizedMatrix, unpack_matrix
from .utils import quantized_params_to_shaped_arrays

try:
    from .op import quantized_matmul
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False


leaf_pred = lambda x: isinstance(x, QuantizedMatrix)
custom_flatten = partial(jax.tree_util.tree_flatten, is_leaf=leaf_pred)

def partial_static_args(f, static_argnums, *args):
    outer_args = args

    @wraps(f)
    def inner(*args, **kwargs):
        outer_arg_iter = (outer_args[i] for i in static_argnums)
        inner_arg_iter = iter(args)
        args = [next(outer_arg_iter if i in static_argnums else inner_arg_iter) for i in range(len(outer_args))]
        return f(*args, **kwargs)
    return inner

def use_quantized(f, static_argnums=()):
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)
    @wraps(f)
    def inner(*args, **kwargs):
        shape_args, shape_kwargs = quantized_params_to_shaped_arrays((args, kwargs))

        closed_jaxpr = jax.make_jaxpr(f, static_argnums=static_argnums)(*shape_args, **shape_kwargs)

        f_with_static_args = partial_static_args(f, static_argnums, *args)

        dynamic_shape_args = [arg for i, arg in enumerate(shape_args) if i not in static_argnums]
        output_struct = jax.tree_util.tree_structure(jax.eval_shape(f_with_static_args, *dynamic_shape_args, **shape_kwargs))

        dynamic_args = [arg for i, arg in enumerate(args) if i not in static_argnums]
        flat_args, _ = custom_flatten((dynamic_args, kwargs))
        result = eval_jaxpr_with_quantized_args(
            closed_jaxpr.jaxpr,
            closed_jaxpr.literals,
            *flat_args
        )
        return jax.tree_util.tree_unflatten(output_struct, result)

    return inner

def eval_jaxpr_with_quantized_args(jaxpr, consts, *args):
    def read(v):
      return v.val if isinstance(v, Literal) else env[v]

    def write(v, val):
      env[v] = val

    env = {}
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        args = [*safe_map(read, eqn.invars)]
        ans = None
        if any(isinstance(arg, QuantizedMatrix) for arg in args):
            ans = eval_with_quantized(eqn, args)
            if ans is None:
                args = [unpack_matrix(arg) if isinstance(arg, QuantizedMatrix) else arg for arg in args]

        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        if ans is None:
            ans = eqn.primitive.bind(*subfuns, *args, **bind_params)

        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return safe_map(read, jaxpr.outvars)

def eval_with_quantized(eqn, args):
    if eqn.primitive.name == 'pjit':
        jaxpr = eqn.params['jaxpr']
        literals = jaxpr.literals
        new_fn = partial(eval_jaxpr_with_quantized_args, jaxpr.jaxpr)
        return jax.experimental.pjit.pjit(new_fn)(literals, *args)

    if eqn.primitive.name == 'remat2':
        jaxpr = eqn.params['jaxpr']
        new_fn = partial(eval_jaxpr_with_quantized_args, jaxpr)
        return jax.checkpoint(
            new_fn,
            prevent_cse=eqn.params['prevent_cse'],
            policy=eqn.params['policy']
        )([], *args)

    if eqn.primitive.name != 'dot_general':
       warnings.warn('Only dot_general is supported for now, so quantized matrix will be unpacked')
       return None

    lhs, rhs = args
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params['dimension_numbers']
    if isinstance(lhs, QuantizedMatrix) or rhs_contract != (0,) or lhs_contract != (len(lhs.shape) - 1,):
        warnings.warn('Only X @ W for quantized W is supported, so quantized matrix will be unpacked')
        return None

    if not KERNEL_AVAILABLE:
        warnings.warn('Triton kernel not available, running full unpack followed by matmul')
        return None
    """
    if not jax.tree_util.tree_all(
        jax.tree_map(
            lambda x: x.device_buffer.device().platform == 'gpu',
            (lhs, rhs)
        )
    ):
        warnings.warn('Triton kernel only available on GPU, running full unpack followed by matmul')
        return None
    """

    return quantized_matmul(lhs, rhs)
