#include <cstdio>

template <unsigned int block_size>
__global__
void find_max_task(const float *values, int N, int *max_idx, float *max_value) {
  int m_idx = 0;
  float m_value = -1.0/0.0;
  for (int i = threadIdx.x; i < N; i += block_size) {
    int t_idx = i;
    float t_value = values[i];
    if (m_value > t_value) {
      m_idx = m_idx;
      m_value = m_value;
    } else {
      m_idx = t_idx;
      m_value = t_value;
    }
  }

  // Apply f over all threads of the warp.
  #pragma unroll
  for (int i = 16; i > 0; i /= 2) {
    int t_idx = __shfl_xor_sync(0xFFFFFFFF, m_idx, i);
    float t_value = __shfl_xor_sync(0xFFFFFFFF, m_value, i);
    if (m_value > t_value) {
      m_idx = m_idx;
      m_value = m_value;
    } else {
      m_idx = t_idx;
      m_value = t_value;
    }
  }

  // If a block consists of more than one warp, we need to synchronize across
  // all warps as well.
  if (block_size > 32) {
    __shared__ int temp_idx[32];
    __shared__ float temp_value[32];
    if (threadIdx.x % 32 == 0) {
      temp_idx[threadIdx.x / 32] = m_idx;
      temp_value[threadIdx.x / 32] = m_value;
    }
    __syncthreads();

    // Combine the warp-local results to compute a single global value.
    if (threadIdx.x % 32 < block_size) {
      m_idx = temp_idx[threadIdx.x % 32];
      m_value = temp_value[threadIdx.x % 32];
    } else {
      m_idx = 0;
      m_value = -1.0/0.0;
    }

    #pragma unroll
    for (int i = 16; i > 0; i /= 2) {
      if (block_size >= 64 * i) {
        int t_idx = __shfl_xor_sync(0xFFFFFFFF, m_idx, i);
        float t_value = __shfl_xor_sync(0xFFFFFFFF, m_value, i);
        if (m_value > t_value) {
          m_idx = m_idx;
          m_value = m_value;
        } else {
          m_idx = t_idx;
          m_value = t_value;
        }
      }
    }
    // The first thread of the block writes the agreed upon values to the first
    // index of shared memory.
    if (threadIdx.x == 0) {
      temp_idx[0] = m_idx;
      temp_value[0] = m_value;
    }
    __syncthreads();
    m_idx = temp_idx[0];
    m_value = temp_value[0];
  }

  // Write the resulting values to global memory.
  if (threadIdx.x == 0) {
    max_idx[0] = m_idx;
    max_value[0] = m_value;
  }
}

extern "C"
void find_max(float *values, int N, int *max_idx, float *max_value) {
  find_max_task<512><<<1, 512>>>(values, N, max_idx, max_value);

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
      exit(1);
    }
  }

  {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA ERROR: %s)\n", cudaGetErrorString(err));
      exit(1);
    }
  }
}
