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

# Alternative Triton kernel where we assume the BLOCK_SIZE has been chosen
# based on the maximum number of non-zeros across any row of the input matrix.
# We can therefore omit the loop. However, this version does not perform well
# either - this is most likely because of the variance between rows - the CUDA
# code is flexible in the sense that different blocks can execute different
# amount of code, whereas they all run the same amount (resulting in lots of
# unnecessary computations).
#
# Also, this approach might be rather inefficient in practice if we have
# matrices with varying number of max non-zeros per row, because the kernel
# would need to be recompiled for every block size.
@triton.jit
def spmv_triton_precompute_kernel(
    A_values, A_rows, A_cols, x_ptr, y_ptr, BLOCK_SIZE : tl.constexpr):
  row_idx = tl.program_id(axis=0)
  curr_row_idx = tl.load(A_rows + row_idx)
  next_row_idx = tl.load(A_rows + row_idx + 1)
  indices = curr_row_idx + tl.arange(0, BLOCK_SIZE)
  mask = indices < next_row_idx
  values = tl.load(A_values + indices, mask=mask)
  cols = tl.load(A_cols + indices, mask=mask)
  x = tl.load(x_ptr + cols, mask=mask)
  s = tl.sum(values * x)
  tl.store(y_ptr + row_idx, s)

def spmv_triton_precompute_wrap(A, x, block_sz):
  N = len(A.crow_indices()) - 1
  y = torch.empty(N, dtype=torch.float32, device='cuda', requires_grad=False)
  grid = lambda meta: (N, )
  spmv_triton_precompute_kernel[grid](A.values(), A.crow_indices(), A.col_indices(), x, y, BLOCK_SIZE=block_sz)
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

# Sanity check (GPU)
y_torch = A.matmul(x)
start.record()
for i in range(10):
  y_torch = A.matmul(x)
end.record()
torch.cuda.synchronize()
print(f"Torch (GPU) {start.elapsed_time(end)} ms")

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
  print(f"Triton1[{block_sz}] {start.elapsed_time(end)} ms")
  assert torch.allclose(y_torch, y_triton, atol=1e-5), f"{y_torch} != {y_triton}"

# Triton (with precomputed max non-zeroes across any row)

# Precompute the maximum number of non-zero elements in any row, and
# determine the block size based on this.
max_nnz = max([y - x for x, y in zip(A.crow_indices()[:-1], A.crow_indices()[1:])])
block_sz = int(triton.next_power_of_2(max_nnz))

y_triton2 = spmv_triton_precompute_wrap(A, x, block_sz)
start.record()
for i in range(10):
  y_triton2 = spmv_triton_precompute_wrap(A, x, block_sz)
end.record()
torch.cuda.synchronize()
print(f"Triton2 {start.elapsed_time(end)} ms")

# PyTorch on CPU
A.cpu()
x.cpu()
y_torch_cpu = A.matmul(x)
t0 = time.time_ns()
for i in range(10):
  y_torch_cpu = A.matmul(x)
t1 = time.time_ns()
print(f"Torch (CPU) {(t1-t0)/1e6} ms")
y_torch_cpu.cuda()

assert torch.allclose(y_torch, y, atol=1e-5), f"{y_torch} != {y}"
assert torch.allclose(y_torch, y_torch_cpu, atol=1e-5), f"{y_torch} != {y_torch_cpu}"
