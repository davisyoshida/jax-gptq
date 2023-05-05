from collections import defaultdict
from copy import deepcopy
from functools import partial

import jax
from jax._src.core import Literal
from jax.util import safe_map
from tqdm import tqdm

from .gptq import gptq, pack_matrix

def tree_size_bytes(tree):
    return jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(
            lambda x: x.size * x.itemsize,
            tree
        ),
        0
    )

#jit_gptq = jax.jit(gptq, static_argnames=('block_size',))
def quantize(fn, params, inputs, block_size=128, actorder=False, damping=0.01):
    closed_jaxpr = jax.make_jaxpr(fn)(params, inputs[0])
    params = jax.device_put(params, jax.devices('cpu')[0])
    inputs = jax.device_put(inputs, jax.devices('cpu')[0])

    argnums = set()
    param_args, param_struct = jax.tree_util.tree_flatten(params)
    input_args = [jax.tree_util.tree_leaves(inp) for inp in inputs]
    input_args = [list(arg) for arg in zip(*input_args)]

    argnums = set(range(0, len(param_args)))

    result = _eval_and_quantize(
        closed_jaxpr.jaxpr,
        closed_jaxpr.literals,
        argnums,
        *param_args,
        *input_args,
        block_size=block_size,
        actorder=actorder,
        damping=damping
    )
    for ind, quantized_param in result.items():
        param_args[ind] = pack_matrix(*quantized_param)

    return jax.tree_util.tree_unflatten(param_struct, param_args)

def _get_delete_points(jaxpr):
    deps = defaultdict(set)
    for i, eqn in enumerate(jaxpr.eqns):
        for var in set(v for v in eqn.invars if not isinstance(v, Literal)):
            deps[var].add(i)

    deps = dict(deps)
    delete_vars = []
    for i, eqn in enumerate(jaxpr.eqns):
        eqn_delete = []
        for var in set(v for v in eqn.invars if not isinstance(v, Literal)):
            deps[var].remove(i)
            if not deps[var]:
                eqn_delete.append(var)
                del deps[var]
        delete_vars.append(eqn_delete)
    return delete_vars

def _eval_and_quantize(jaxpr, consts, argnums, *args, block_size=128, actorder=False, damping=0.01):
    cpu = jax.devices('cpu')[0]
    gpu = jax.devices('gpu')[0]
    # Args are all either params or lists of tensors

    quantized_results = {}
    name_to_pos = {}

    n_batches = len(next(a for i, a in enumerate(args) if i not in argnums))
    envs = [{} for _ in range(n_batches)] # Everything in here should be on GPU
    param_env = {} # Only some things in here should be on GPU

    for index, name in enumerate(jaxpr.invars):
        if index in argnums:
            param_env[name] = args[index]
            name_to_pos[name] = index
        else:
            for i in range(n_batches):
                envs[i][name] = args[index][i]

    def delete(name):
        if name not in envs[0]:
            return
        for env in envs:
            env[name].device_buffer.delete()
            del env[name]

    delete_points = _get_delete_points(jaxpr)

    env = defaultdict(list)
    const_env = {name: val for name, val in zip(jaxpr.constvars, consts)}
    pos = 0
    bar = tqdm()
    while True:
        bar.update(1)
        seek_vars = {name for name, val in param_env.items() if len(val.shape) == 2}
        next_pos, needed_names = find_next_matmul(jaxpr, pos, seek_vars)
        if next_pos is None:
            break

        block_param_env = {
            name: jax.device_put(param_env[name], gpu)
            for name in needed_names if name in param_env
        }

        #print(f'Current env size: {tree_size_bytes(envs):.2e} bytes')
        #print(f'Current param env size: {tree_size_bytes(block_param_env):.2e} bytes')

        delete_keys = set(var for i in range(pos, next_pos) for var in delete_points[i])

        block_fn = jax.jit(partial(run_segment, jaxpr, pos, next_pos, delete_points))
        for i, env in enumerate(envs):
            gpu_env = jax.device_put(env, gpu)
            new_env = block_fn(block_param_env, gpu_env, const_env)
            envs[i] = jax.device_put(new_env, cpu)
            def maybe_delete(val):
                if not val.is_deleted():
                    val.device_buffer.delete()
            jax.tree_map(maybe_delete, (gpu_env, new_env))

        for param in block_param_env.values():
            param.device_buffer.delete()
        del block_param_env

        #(jax.device_put(0., gpu) + 0).block_until_ready()

        matmul_eqn = jaxpr.eqns[next_pos]
        lhs_name, w_name = matmul_eqn.invars

        xs = []
        for env in envs:
            lhs_val = jax.device_put(env[lhs_name], gpu)
            xs.append(lhs_val)


        w = jax.device_put(param_env[w_name], gpu)
        quantized_w, quantize_params = gptq(w, xs, block_size=block_size, actorder=actorder, damping=damping)

        assert quantized_w.shape == w.shape
        outvar, = jaxpr.eqns[next_pos].outvars
        matmul_eval = jax.jit(partial(eval_eqn, matmul_eqn))
        #matmul_eval = partial(eval_eqn, matmul_eqn)
        for env in envs:
            #env[outvar] = eval_eqn(matmul_eqn, env[lhs_name], quantized_w)
            gpu_lhs = jax.device_put(env[lhs_name], gpu)
            env[outvar] = matmul_eval(gpu_lhs, quantized_w)
            gpu_lhs.device_buffer.delete()
            #(jax.device_put(0., gpu) + 0).block_until_ready()

        for name in delete_points[next_pos]:
            delete(name)

        param_env[w_name].device_buffer.delete()

        cpu_quantized_w = jax.device_put(quantized_w, cpu)
        quantized_w.device_buffer.delete()

        param_env[w_name] = cpu_quantized_w
        quantized_results[name_to_pos[w_name]] = cpu_quantized_w, jax.device_put(quantize_params, cpu)

        #(jax.device_put(0., gpu) + 0).block_until_ready()

        pos = next_pos + 1
    return quantized_results

def find_next_matmul(jaxpr, start_point, target_vars):
    needed_names = set()
    for i, eqn in enumerate(jaxpr.eqns[start_point:], start_point):
        invars = eqn.invars
        if eqn.primitive.name == 'dot_general':
            rhs = invars[1]
            (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params['dimension_numbers']
            if rhs in target_vars and rhs_contract == (0,): # TODO: Might want to check lhs_contract too
                return i, needed_names
        needed_names.update(v for v in invars if not isinstance(v, Literal))
    return None, needed_names

def run_segment(jaxpr, start_pos, next_pos, delete_points, param_env, env, const_env):
    env = dict(env)
    def read(v):
        if isinstance(v, Literal):
            return v.val
        if v in param_env:
            return param_env[v]
        if v in env:
            return env[v]
        return const_env[v]


    def write(v, val):
        env[v] = val

    for i, eqn in enumerate(jaxpr.eqns[start_pos:next_pos], start_pos):
        eqn_args = safe_map(read, eqn.invars)
        ans = eval_eqn(eqn, *eqn_args)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)

        for varname in delete_points[i]:
            if varname in env:
                del env[varname]
    return env

def eval_eqn(eqn, *args):
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    ans = eqn.primitive.bind(*subfuns, *args, **bind_params)
    return ans

# State:
# What point am I at? Either start or a matmul
# Env for evaluating that segment

# Info needed:
#   - What can I free
#   - How many batches are there

# Other:
# Move args for segment to GPU
