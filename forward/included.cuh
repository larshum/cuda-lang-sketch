// This file is provided by the user.

#include "forward.cuh"

__device__
float initial_prob(HMM hmm, uint32_t x) {
  return hmm.initial_prob[x];
}
template <int num_states>
__device__
float output_prob(HMM hmm, uint32_t x, uint8_t o) {
  return hmm.output_prob[o * (num_states / 16) + x % (num_states / 16)];
}
template <int num_states>
__device__
float transition_prob(HMM hmm, uint32_t x, uint32_t y) {
  return hmm.trans1[(x % (num_states / 16)) * 4 + y % 4] + hmm.trans2[y / (num_states / 16)];
}
__device__
float transition_prob1(HMM hmm, uint32_t x, uint32_t y) {
  return hmm.gamma;
}
__device__
float transition_prob2(HMM hmm, uint32_t x, uint32_t y) {
  return hmm.synthetic_248;
}
__device__
float transition_prob3(HMM hmm, uint32_t x, uint32_t y) {
  return 0.;
}

template <int num_states>
__device__
int64_t forward_prob_predecessors(
    HMM hmm, float *alpha_prev, int64_t instance, uint32_t state,
    float *probs) {
  int64_t pidx = 0;
  uint32_t pred;
  float p;
  {
    uint32_t x1 = 0;
    while ((x1 < 4)) {
      pred = (x1 * (num_states / 64)) + (state / 4) % (num_states / 64);
      (p = transition_prob<num_states>(hmm, pred, state));
      {
        ((probs[pidx]) = (p + (alpha_prev[((instance * num_states) + pred)])));
        (pidx = (pidx + 1));
      }
      (x1 = (x1 + 1));
    }
  }
  {
    if (((state / (num_states / 16)) == 15)) {
      {
        (pred = state);
        (p = transition_prob1(hmm, pred, state));
        ;
      }
    } else {
      
    }
    if (((state / (num_states / 16)) == 14)) {
      {
        (pred = ((((state / (num_states / 16)) + 1) * (num_states / 16)) + (state % (num_states / 16))));
        (p = transition_prob2(hmm, pred, state));
        ;
      }
    } else {
      
    }
    if ((((state / (num_states / 16)) != 14) && ((state / (num_states / 16)) != 15))) {
      {
        (pred = ((((state / (num_states / 16)) + 1) * (num_states / 16)) + (state % (num_states / 16))));
        (p = transition_prob3(hmm, pred, state));
        ;
      }
    } else {
      
    }
    {
      ((probs[pidx]) = (p + (alpha_prev[((instance * num_states) + pred)])));
      (pidx = (pidx + 1));
    }
  }
  return pidx;
}

__device__
float log_sum_exp(float* probs, float neginf) {
  float maxp = probs[0];
  for (int i = 1; i < 5; i++) {
    if (probs[i] > maxp) maxp = probs[i];
  }
  if (maxp == neginf) return maxp;
  float sum = 0.0;
  for (int i = 0; i < 5; i++) {
    sum += expf(probs[i] - maxp);
  }
  return maxp + logf(sum);
}
