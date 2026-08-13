import tilelang
import tilelang.language as T
from tilelang import jit
import torch

@jit
def matmul(
    M: int, N: int, K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float32"
):
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype), #type: ignore 
        B: T.Tensor((K, N), dtype), #type: ignore
        C: T.Tensor((M, N), dtype), #type: ignore
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128
        ) as (bx, by):

            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)                   

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


M, N, K = 1024, 1024, 1024

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)

kernel = matmul(M, N, K)
kernel(A, B, C)

C_ref = A @ B
print("Max error:", (C - C_ref).abs().max().item())
print("Matrix multiplication successful!")