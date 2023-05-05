import triton
import triton.language as tl

@triton.autotune(
    configs=[
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        #triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K']#, 'out_type']
)
@triton.jit
def matmul_4bit_quantized(
    # Pointers to inputs
    a_ptr, b_ptr, zeros_ptr, scales_ptr,
    # Pointers to outputs
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Adapted from from https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
    Kernel for computing the matmul C = A x B, where B is an int32 matrix with each value consisting of 8 packed 4 bit values.
    A has shape (M, K), B has shape (K // 8, N) and C has shape (M, N)
    """
    VALS_PER_INT : tl.constexpr = 2
    BLOCK_SIZE_BK : tl.constexpr = BLOCK_SIZE_K // VALS_PER_INT

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K // 8, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_BK) * VALS_PER_INT
    offs_bk = tl.arange(0, BLOCK_SIZE_BK)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Load zeros and scales
    # Probably don't need to mask out of bounds vals since they should get masked in the matmul calc
    block_zeros = tl.load(zeros_ptr + offs_bn[None, :])
    block_scales = tl.load(scales_ptr + offs_bn[None, :])

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32,)

    BK = K // VALS_PER_INT

    for block_num in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b = tl.load(b_ptrs, mask=offs_bk[:, None] < BK - block_num * BLOCK_SIZE_BK, other=0)

        a_subptrs = a_ptrs

        for i in range(VALS_PER_INT):
            b_slice = (b & 0xF).to(tl.float32)
            b_slice -= block_zeros
            b_slice *= block_scales

            a_slice = tl.load(
                a_subptrs,
                mask=offs_k[None, :] < K - block_num * BLOCK_SIZE_K - i,
                other=0.
            ).to(tl.float32)

            b = (b >> 4).to(tl.uint8)
            a_subptrs += stride_ak

            accumulator += tl.dot(a_slice, b_slice)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_BK * stride_bk

    c = accumulator.to(tl.float16)
    """
    if out_type == 'float16':
        c = accumulator.to(tl.float16)
    elif out_type == 'float32':
        c = accumulator
    else:
        raise ValueError(f'Only float16 and float32 supported for out_type, got {out_type}')
    """

    # -----------------------------------------------------------
    # Write back the block of the output matrix C wth masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.autotune(
    configs=[
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        #triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        #triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K']#, 'out_type']
)
@triton.jit
def matmul_4bit_quantized_traponse_b(
    # Pointers to inputs
    a_ptr, b_ptr, zeros_ptr, scales_ptr,
    # Pointers to outputs
    c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Adapted from from https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py
    Kernel for computing the matmul C = A x B, where B is an int32 matrix with each value consisting of 8 packed 4 bit values.
    A has shape (M, K), B has shape (N, K) and C has shape (M, 2N)
    """

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K // 8, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)


    zero_ptrs = zeros_ptr + offs_k[:, None]
    scale_ptrs = scales_ptr + offs_k[:, None]

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        transposed_k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=transposed_k_mask, other=0.0)

        block_zeros =  tl.load(zero_ptrs, mask=transposed_k_mask, other=0.)
        block_scales = tl.load(scale_ptrs, mask=transposed_k_mask, other=1.)

        b1_slice = ((b & 0xF).to(tl.float32) - block_zeros) * block_scales
        b2_slice = ((b >> 4 ).to(tl.float32) - block_zeros) * block_scales

        # We accumulate along the K dimension.
        #accumulator1 += tl.sum(a[:, None, :] * b1_slice[None, :, :], axis=2)
        #accumulator2 += tl.sum(a[:, None, :] * b2_slice[None, :, :], axis=2)
        accumulator1 += tl.dot(a, b1_slice)
        accumulator2 += tl.dot(a, b2_slice)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

        zero_ptrs += BLOCK_SIZE_K
        scale_ptrs += BLOCK_SIZE_K

    c1 = accumulator1.to(tl.float16)
    c2 = accumulator2.to(tl.float16)
    """
    if out_type == 'float16':
        c = accumulator.to(tl.float16)
    elif out_type == 'float32':
        c = accumulator
    else:
        raise ValueError(f'Only float16 and float32 supported for out_type, got {out_type}')
    """

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    offs_cn = 2 * (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < 2 * N)
    tl.store(c_ptrs, c1, mask=c_mask)

    offs_cn += 1
    c_ptrs2 = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask2 = (offs_cm[:, None] < M) & (offs_cn[None, :] < 2 * N)
    tl.store(c_ptrs2, c2, mask=c_mask2)
