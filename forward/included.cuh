// This file is provided by the user.

#include "forward.cuh"

__device__
float transition_prob(HMM hmm, uint16_t x, uint16_t y) {
  return ((hmm.trans1[(((x % 64) * 4) + (y % 4))]) + (hmm.trans2[(y / 64)]));
}
__device__
float transition_prob1(HMM hmm, uint16_t x, uint16_t y) {
  return hmm.gamma;
}
__device__
float transition_prob2(HMM hmm, uint16_t x, uint16_t y) {
  return hmm.synthetic_248;
}
__device__
float transition_prob3(HMM hmm, uint16_t x, uint16_t y) {
  return 0.;
}

__device__
int64_t forward_prob_predecessors(
    HMM hmm, float *alpha_prev, int64_t instance, uint16_t state,
    float *probs) {
  int64_t pidx = 0;
  uint16_t pred;
  float p;
  {
    uint16_t x1 = 0;
    while ((x1 < 4)) {
      {   
        (pred = (((0 * 64) + (x1 * 16)) + ((state / 4) % 16)));
        (p = transition_prob(hmm, pred, state));
        {
          ((probs[pidx]) = (p + (alpha_prev[((instance * 1024) + pred)])));
          (pidx = (pidx + 1));
        }
      }
      (x1 = (x1 + 1));
    }
  }
  {
    if (((state / 64) == 15)) {
      {
        (pred = state);
        (p = transition_prob1(hmm, pred, state));
        ;
      }
    } else {
      
    }
    if (((state / 64) == 14)) {
      {
        (pred = ((((state / 64) + 1) * 64) + (state % 64)));
        (p = transition_prob2(hmm, pred, state));
        ;
      }
    } else {
      
    }
    if ((((state / 64) != 14) && ((state / 64) != 15))) {
      {
        (pred = ((((state / 64) + 1) * 64) + (state % 64)));
        (p = transition_prob3(hmm, pred, state));
        ;
      }
    } else {
      
    }
    {
      ((probs[pidx]) = (p + (alpha_prev[((instance * 1024) + pred)])));
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
