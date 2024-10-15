// When we define types inside the IR, this is generated in a separate header
// file. This allows us to pass those types into external functions.

#pragma once

#include <stdint.h>

struct HMM {
  float gamma;
  float *trans1;
  float *trans2;
  float *output_prob;
  float *initial_prob;
  float synthetic_248;
};

struct ObsSeqs {
  uint8_t *data;
  int64_t *lens;
  int64_t maxlen;
  int64_t num_instances;
};
