import torch
import time


torch.manual_seed(42)

N = 512
REPEAT = 20

# 使用 FP64 构造参考数据
A_ref = torch.randn(N, N, dtype=torch.float64)
B_ref = torch.randn(N, N, dtype=torch.float64)

# FP64 结果作为高精度参考
C_ref = A_ref @ B_ref


def benchmark(dtype):
    A = A_ref.to(dtype)
    B = B_ref.to(dtype)

    # warm up
    for _ in range(3):
        C = A @ B

    start = time.perf_counter()

    for _ in range(REPEAT):
        C = A @ B

    end = time.perf_counter()

    avg_time = (end - start) / REPEAT

    # 转回 FP64 再计算误差
    C = C.to(torch.float64)

    relative_error = (
        torch.linalg.vector_norm(C - C_ref)
        / torch.linalg.vector_norm(C_ref)
    )

    max_error = torch.max(torch.abs(C - C_ref))

    return avg_time, relative_error.item(), max_error.item()


dtypes = [
    torch.float32,
    torch.float16,
    torch.bfloat16
]


for dtype in dtypes:
    try:
        avg_time, relative_error, max_error = benchmark(dtype)

        print("=" * 40)
        print("dtype:", dtype)
        print(f"average time: {avg_time * 1000:.3f} ms")
        print(f"relative error: {relative_error:.6e}")
        print(f"max absolute error: {max_error:.6e}")

    except RuntimeError as e:
        print("=" * 40)
        print(dtype, "当前 CPU 不支持或没有优化该操作")
        print(e)