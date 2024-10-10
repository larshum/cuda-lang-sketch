#include <cstdint>

struct CSRMatrix {
  float *values;
  int64_t *rows;
  int64_t *cols;
  int64_t nnz;
  int64_t nrows;
  int64_t ncols;
};

template <int64_t block_size>
__global__
void spmv_row(CSRMatrix A, float *x, float *y) {
  int64_t row = blockIdx.x;
  float sum = 0.0;
  // Compute the sum of all values in parallel over all threads of the block.
  for (int64_t i = A.rows[row] + threadIdx.x; i < A.rows[row+1]; i += block_size) {
    sum = sum + A.values[i] * x[A.cols[i]];
  }

  // Sum over all threads of a warp.
  sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 16);
  sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 8);
  sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 4);
  sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 2);
  sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 1);

  // We assume blockSize is always a multiple of 32. If we use more than one
  // warp, we use shared memory to distribute the sum among all warps.
  if (block_size > 32) {
    // The first thread of each warp writes the warp-local sums to shared memory.
    __shared__ float temp[32];
    if (threadIdx.x % 32 == 0) {
      temp[threadIdx.x / 32] = sum;
    }
    __syncthreads();

    // Sum together the warp-local sums to compute the global sum.
    sum = temp[threadIdx.x % 32];
    if (block_size == 1024) sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    if (block_size >= 512) sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    if (block_size >= 256) sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    if (block_size >= 128) sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    if (block_size >= 64) sum = sum + __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    if (threadIdx.x == 0) {
      temp[0] = sum;
    }
    __syncthreads();
    sum = temp[0];
  }

  // Write the resulting sum to global memory.
  if (threadIdx.x == 0) {
    y[row] = sum;
  }
}

extern "C"
void spmv(
    float *A_values, int64_t *A_rows, int64_t *A_cols,
    int64_t A_nnz, int64_t A_nrows, int64_t A_ncols,
    float *x, float *y) {
  // Construct the CSRMatrix as declared in the specification. We unwrap it to
  // its constituents when passing it across the C API to avoid having to
  // construct it on the other end (where the declared struct type is not
  // available).
  CSRMatrix A;
  A.values = A_values;
  A.rows = A_rows;
  A.cols = A_cols;
  A.nnz = A_nnz;
  A_nrows = A_nrows;
  A_ncols = A_ncols;
  {
    // The user can easily modify this value, either directly in the generated
    // code or by annotating the task set launch in the IR code.
    const int64_t warps_per_block = 1;
    const int64_t block_size = warps_per_block * 32;

    // Launching a set of tasks (typically) results in a kernel launch.
    //spmv_row<block_size><<<A_nrows, block_size>>>(A_values, A_rows, A_cols, A_nnz, A_nrows, A_ncols, x, y);
    spmv_row<block_size><<<A_nrows, block_size>>>(A, x, y);
  }
}
