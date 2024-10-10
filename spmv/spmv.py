import numpy as np
import time
import torch
import triton
import triton.language as tl

from spmv_wrap import CSRMatrix, SpmvWrapper

# Implementation of row-by-row SpMV in Triton
@triton.jit
def spmv_triton_kernel(
    A_values, A_rows, A_cols, x_ptr, y_ptr, BLOCK_SIZE : tl.constexpr):
  row_idx = tl.program_id(axis=0)

  curr_row_idx = tl.load(A_rows + row_idx)
  next_row_idx = tl.load(A_rows + row_idx + 1)

  # The number of non-zero elements is not considered constant to Triton (it is
  # not known when the code is JIT compiled), but the arguments passed to
  # 'arange' must be constant. To work around this, we break up the computation
  # in a for-loop processing a constant number of elements per iteration.
  #
  # In the native CUDA implementation, we can sum all values locally in each
  # thread before performing a parallel reduction. In this version, however,
  # we have to do a reduction per block we sum over, which is noticeably less
  # efficient.
  s = 0.0
  for idx in range(curr_row_idx, next_row_idx, BLOCK_SIZE):
    indices = idx + tl.arange(0, BLOCK_SIZE)
    mask = indices < next_row_idx
    values = tl.load(A_values + indices, mask=mask)
    cols = tl.load(A_cols + indices, mask=mask)
    x = tl.load(x_ptr + cols, mask=mask)
    s += tl.sum(values * x)

  tl.store(y_ptr + row_idx, s)

def spmv_triton_wrap(A, x, block_sz):
  N = len(A.crow_indices()) - 1
  y = torch.empty(N, dtype=torch.float32, device='cuda', requires_grad=False)
  grid = lambda meta: (N, )
  spmv_triton_kernel[grid](A.values(), A.crow_indices(), A.col_indices(), x, y, BLOCK_SIZE=block_sz)
  return y

# Ignore the Sparse CSR warning
import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

def uniform_random_csr_f32_i64(N, M, d):
  nnz = int(d * N * M)
  s = set()
  while len(s) < nnz:
    s.update(np.random.randint(0, N*M, nnz-len(s)))

  flat_idxs = np.array(list(s), dtype=np.int64)
  rows, cols = np.divmod(flat_idxs, M)
  values = torch.randn(nnz, dtype=torch.float32, device='cuda')
  idxs = np.array((rows, cols))
  A = torch.sparse_coo_tensor(idxs, values, device='cuda', requires_grad=False, check_invariants=True)
  return A.to_sparse_csr()

N = 4096
M = 16384
sparsity = 0.01
A = uniform_random_csr_f32_i64(N, M, sparsity)
x = torch.randn(M, dtype=torch.float32, device='cuda', requires_grad=False)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Sanity check
y_torch = A.matmul(x)
start.record()
for i in range(10):
  y_torch = A.matmul(x)
end.record()
torch.cuda.synchronize()
print(f"Torch {start.elapsed_time(end)} ms")

# Calling the "generated" code from our IR.
A_cu = CSRMatrix(A.values(), A.crow_indices(), A.col_indices(), len(A.values()), N, M)
sw = SpmvWrapper()
y = torch.empty(N, dtype=torch.float32, device='cuda', requires_grad=False)
sw.spmv(A_cu, x, y)
start.record()
for i in range(10):
  y = torch.empty(N, dtype=torch.float32, device='cuda', requires_grad=False)
  sw.spmv(A_cu, x, y)
end.record()
torch.cuda.synchronize()
print(f"CUDA IR {start.elapsed_time(end)} ms")

# Triton
for block_sz in [2**x for x in range(1, 12)]:
  y_triton = spmv_triton_wrap(A, x, block_sz)
  start.record()
  for i in range(10):
    y_triton = spmv_triton_wrap(A, x, block_sz)
  end.record()
  torch.cuda.synchronize()
  print(f"Triton[{block_sz}] {start.elapsed_time(end)} ms")

assert torch.allclose(y_torch, y_triton, atol=1e-5), f"{y_torch} != {y_triton}"
assert torch.allclose(y, y_torch, atol=1e-5), f"{y} != {y_torch}"
