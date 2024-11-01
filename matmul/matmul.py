import numpy as np
import time

import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

N = 4096
M = 4096
sparsity = 0.01

import scorch as torch
def uniform_random_csr_f32_i64(N, M, d):
  nnz = int(d * N * M)
  s = set()
  while len(s) < nnz:
    s.update(np.random.randint(0, N*M, nnz-len(s)))

  flat_idxs = np.array(list(s), dtype=np.int64)
  rows, cols = np.divmod(flat_idxs, M)
  values = torch.randn(nnz, dtype=torch.float32)
  idxs = np.array((rows, cols))
  A = torch.sparse_coo_tensor(idxs, values)
  return A.to_sparse_csr()

A = uniform_random_csr_f32_i64(N, M, sparsity)
B = uniform_random_csr_f32_i64(N, M, sparsity)

y = torch.matmul(A, B)
t0 = time.time_ns()
for i in range(10):
  y = torch.matmul(A, B)
t1 = time.time_ns()
print(f"Scorch (CPU) {(t1-t0)/1e6} ms")

import torch
def uniform_random_csr_f32_i64(N, M, d):
  nnz = int(d * N * M)
  s = set()
  while len(s) < nnz:
    s.update(np.random.randint(0, N*M, nnz-len(s)))

  flat_idxs = np.array(list(s), dtype=np.int64)
  rows, cols = np.divmod(flat_idxs, M)
  values = torch.randn(nnz, dtype=torch.float32)
  idxs = np.array((rows, cols))
  A = torch.sparse_coo_tensor(idxs, values)
  return A.to_sparse_csr()

A = uniform_random_csr_f32_i64(N, M, sparsity)
B = uniform_random_csr_f32_i64(N, M, sparsity)

y = torch.matmul(A, B)
t0 = time.time_ns()
for i in range(10):
  y = torch.matmul(A, B)
t1 = time.time_ns()
print(f"Torch (CPU) {(t1-t0)/1e6} ms")

def uniform_random_csr_f32_i64(N, M, d):
  nnz = int(d * N * M)
  s = set()
  while len(s) < nnz:
    s.update(np.random.randint(0, N*M, nnz-len(s)))

  flat_idxs = np.array(list(s), dtype=np.int64)
  rows, cols = np.divmod(flat_idxs, M)
  values = torch.randn(nnz, dtype=torch.float32, device='cuda')
  idxs = np.array((rows, cols))
  A = torch.sparse_coo_tensor(idxs, values, device='cuda')
  return A.to_sparse_csr()

A = uniform_random_csr_f32_i64(N, M, sparsity)
B = uniform_random_csr_f32_i64(N, M, sparsity)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
y_gpu = torch.matmul(A, B)
t0 = time.time_ns()
for i in range(10):
  y_gpu = torch.matmul(A, B)
torch.cuda.synchronize()
t1 = time.time_ns()
print(f"Torch (GPU) {(t1-t0)/1e6} ms")
