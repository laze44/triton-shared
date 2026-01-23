import torch

import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Compute C = A x B, where A:(M,K), B:(K,N), C:(M,N)."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < K - k * BLOCK_SIZE_K
        mask_am = offs_am < M
        mask_bn = offs_bn < N
        mask_a = tl.broadcast_to(mask_am[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_K)) & \
            tl.broadcast_to(mask_k[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_K))
        mask_b = tl.broadcast_to(mask_k[:, None], (BLOCK_SIZE_K, BLOCK_SIZE_N)) & \
            tl.broadcast_to(mask_bn[None, :], (BLOCK_SIZE_K, BLOCK_SIZE_N))
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_cm = offs_cm < M
    mask_cn = offs_cn < N
    c_mask = tl.broadcast_to(mask_cm[:, None], (BLOCK_SIZE_M, BLOCK_SIZE_N)) & \
        tl.broadcast_to(mask_cn[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=16,
    )
    return c


def test_matmul(device="cpu"):
    torch.manual_seed(0)
    a = torch.randn((128, 256), device=device, dtype=torch.float32)
    b = torch.randn((256, 64), device=device, dtype=torch.float32)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    torch.testing.assert_close(triton_output, torch_output, atol=1e-2, rtol=0)


if __name__ == "__main__":
    triton.runtime.driver.set_active(CPUDriver())
    test_matmul()
    print("simple_matmul: OK")
