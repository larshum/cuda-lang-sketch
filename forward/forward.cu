#include "forward.cuh"
#include <cstdio>

#include "included.cuh"
__device__
extern float log_sum_exp(float *probs, float neginf);

__device__
extern int64_t forward_prob_predecessors(HMM hmm, float *alpha_prev, int64_t instance, uint16_t state, float *probs);

template <unsigned int block_size>
__global__
void forward_init(HMM hmm, ObsSeqs seqs, float *alpha_src) {
  int64_t task_index = blockIdx.x;
  int64_t instance = task_index;
  uint8_t o = seqs.data[instance * seqs.maxlen];
  for (int64_t state = threadIdx.x; state < 1024; state += block_size) {
    alpha_src[instance * 1024 + state] = hmm.initial_prob[state] + hmm.output_prob[o * 64 + state % 64];
  }
}

template <unsigned int block_size>
__global__
void forward_step(HMM hmm, ObsSeqs seqs, float *alpha1, float *alpha2, int64_t t, float neginf) {
  int64_t task_index = blockIdx.x;
  int64_t instance = task_index;
  float *alpha_src, *alpha_dst;
  if (t & 1) {
    alpha_src = alpha1;
    alpha_dst = alpha2;
  } else {
    alpha_src = alpha2;
    alpha_dst = alpha1;
  }
  for (int64_t state = threadIdx.x; state < 1024; state += block_size) {
    int64_t idx = instance * 1024 + state;
    if (t < seqs.lens[instance]) {
      uint8_t o = seqs.data[instance * seqs.maxlen + t];
      float probs[5];
      int64_t pidx = forward_prob_predecessors(hmm, alpha_src, instance, state, probs);
      while (pidx < 5) {
        probs[pidx] = neginf;
        pidx = pidx + 1;
      }
      alpha_dst[idx] = log_sum_exp(probs, neginf) + hmm.output_prob[o * 64 + state % 64];
    } else if (t == seqs.lens[instance]) {
      alpha_dst[idx] = alpha_src[idx];
    }
  }
}

template <unsigned int block_size>
__global__
void forward_steps(HMM hmm, ObsSeqs seqs, float *alpha1, float *alpha2, float neginf) {
  int64_t task_index = blockIdx.x;
  int64_t instance = task_index;
  for (int64_t t = 1; t < seqs.maxlen; t++) {
    float *alpha_src, *alpha_dst;
    if (t & 1) {
      alpha_src = alpha1;
      alpha_dst = alpha2;
    } else {
      alpha_src = alpha2;
      alpha_dst = alpha1;
    }
    for (int64_t state = threadIdx.x; state < 1024; state += block_size) {
      int64_t idx = instance * 1024 + state;
      if (t < seqs.lens[instance]) {
        uint8_t o = seqs.data[instance * seqs.maxlen + t];
        float probs[5];
        int64_t pidx = forward_prob_predecessors(hmm, alpha_src, instance, state, probs);
        while (pidx < 5) {
          probs[pidx] = neginf;
          pidx = pidx + 1;
        }
        alpha_dst[idx] = log_sum_exp(probs, neginf) + hmm.output_prob[o * 64 + state % 64];
      } else if (t == seqs.lens[instance]) {
        alpha_dst[idx] = alpha_src[idx];
      }
    }
    // At the end of each iteration of the sequential outer loop, we need to
    // synchronize to ensure all results of the current iteration are available
    // at the start of the next iteration. The IR compiler has to insert this
    // to ensure the correct execution order.
    //
    // We synchronize as few threads as possible to reduce the overhead. In
    // particular, when we have at most 32 threads, we only need to sync a
    // single warp of threads (which should be faster than synchronizing
    // multiple threads).
    block_size > 32 ? __syncthreads() : __syncwarp();
  }
}

template <unsigned int block_size>
__global__
void forward_lse(HMM hmm, ObsSeqs seqs, float *alpha1, float *alpha2, float neginf, float *result) {
  int64_t task_index = blockIdx.x;
  float *alpha;
  if (seqs.maxlen & 1) {
    alpha = alpha1;
  } else {
    alpha = alpha2;
  }

  int64_t ofs = task_index * 1024;
  float maxp = neginf;
  for (int state = threadIdx.x; state < 1024; state += block_size) {
    maxp = max(maxp, alpha[ofs + state]);
  }

  // Warp-level synchronization.
  for (int i = 16; i > 0; i /= 2) {
    maxp = max(maxp, __shfl_xor_sync(0xFFFFFFFF, maxp, i));
  }

  // Block-level synchronization.
  __shared__ float temp[32];
  if (threadIdx.x % 32 == 0) {
    temp[threadIdx.x / 32] = maxp;
  }
  __syncthreads();
  if (threadIdx.x % 32 < block_size / 32) {
    maxp = temp[threadIdx.x % 32];
  } else {
    maxp = neginf;
  }

  for (int i = 16; i > 0; i /= 2) {
    if (block_size >= 64 * i) {
      maxp = max(maxp, __shfl_xor_sync(0xFFFFFFFF, maxp, i));
    }
  }

  // First thread writes result to index 0, which all threads read from after
  // synchronization to make them agree upon a single value.
  if (threadIdx.x == 0) {
    temp[0] = maxp;
  }
  __syncthreads();
  maxp = temp[0];

  // Compute the sum of exponentiated probabilities subtracted by the maximum
  // probability.
  float psum = 0.0;
  for (int state = threadIdx.x; state < 1024; state += block_size) {
    psum = psum + expf(alpha[ofs + state] - maxp);
  }

  // Warp-level synchronization.
  for (int i = 16; i > 0; i /= 2) {
    psum = psum + __shfl_xor_sync(0xFFFFFFFF, psum, i);
  }

  // Block-level synchronization.
  if (threadIdx.x % 32 == 0) {
    temp[threadIdx.x / 32] = psum;
  }
  __syncthreads();

  if (threadIdx.x % 32 < block_size / 32) {
    psum = temp[threadIdx.x % 32];
  } else {
    psum = 0.0;
  }

  for (int i = 16; i > 0; i /= 2) {
    if (block_size >= 64 * i) {
      psum = psum + __shfl_xor_sync(0xFFFFFFFF, psum, i);
    }
  }

  // First thread writes result to index 0, which all threads read from after
  // synchronization to make them agree upon a single value.
  if (threadIdx.x == 0) {
    temp[0] = psum;
  }
  __syncthreads();
  psum = temp[0];

  // Write the resulting value to global memory.
  if (threadIdx.x == 0) {
    result[task_index] = maxp + logf(psum);
  }
}

extern "C"
void forward(
    float hmm_gamma, float *hmm_trans1, float *hmm_trans2,
    float *hmm_output_prob, float *hmm_initial_prob, float hmm_synthetic_248,
    int64_t hmm_num_states, uint8_t *seqs_data, int64_t *seqs_lens,
    int64_t seqs_maxlen, int64_t seqs_num_instances, float *result,
    float *alpha1, float *alpha2) {
  HMM hmm;
  hmm.gamma = hmm_gamma;
  hmm.trans1 = hmm_trans1;
  hmm.trans2 = hmm_trans2;
  hmm.output_prob = hmm_output_prob;
  hmm.initial_prob = hmm_initial_prob;
  hmm.synthetic_248 = hmm_synthetic_248;
  ObsSeqs seqs;
  seqs.data = seqs_data;
  seqs.lens = seqs_lens;
  seqs.maxlen = seqs_maxlen;
  seqs.num_instances = seqs_num_instances;

  float neginf = -1.0 / 0.0;
  forward_init<1024><<<seqs.num_instances, 1024>>>(hmm, seqs, alpha1);
  for (int64_t t = 1; t < seqs.maxlen; t++) {
    forward_step<1024><<<seqs.num_instances, 1024>>>(hmm, seqs, alpha1, alpha2, t, neginf);
  }
  forward_lse<512><<<seqs.num_instances, 512>>>(hmm, seqs, alpha1, alpha2, neginf, result);
}

extern "C"
void forward_merged(
    float hmm_gamma, float *hmm_trans1, float *hmm_trans2,
    float *hmm_output_prob, float *hmm_initial_prob, float hmm_synthetic_248,
    int64_t hmm_num_states, uint8_t *seqs_data, int64_t *seqs_lens,
    int64_t seqs_maxlen, int64_t seqs_num_instances, float *result,
    float *alpha1, float *alpha2) {
  HMM hmm;
  hmm.gamma = hmm_gamma;
  hmm.trans1 = hmm_trans1;
  hmm.trans2 = hmm_trans2;
  hmm.output_prob = hmm_output_prob;
  hmm.initial_prob = hmm_initial_prob;
  hmm.synthetic_248 = hmm_synthetic_248;
  ObsSeqs seqs;
  seqs.data = seqs_data;
  seqs.lens = seqs_lens;
  seqs.maxlen = seqs_maxlen;
  seqs.num_instances = seqs_num_instances;

  float neginf = -1.0 / 0.0;
  forward_init<1024><<<seqs.num_instances, 1024>>>(hmm, seqs, alpha1);
  forward_steps<1024><<<seqs.num_instances, 1024>>>(hmm, seqs, alpha1, alpha2, neginf);
  forward_lse<512><<<seqs.num_instances, 512>>>(hmm, seqs, alpha1, alpha2, neginf, result);
}
