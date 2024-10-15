import ctypes
import subprocess

def load_library(lib_id):
  # Below code is included for convenience; it would typically not be generated
  # by the IR compiler as it is beyond its responsibilities (it generates the
  # CUDA code, but it does not know how to run it).
  r = subprocess.run(["make"], capture_output=True)
  if r.returncode != 0:
    print(f"Shared library compilation failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
    exit(r.returncode)
  return ctypes.cdll.LoadLibrary(f"./{lib_id}")

class HMM:
  def __init__(self, gamma, trans1, trans2, output_prob, initial_prob, synthetic_248, num_states):
    self.gamma = gamma
    self.trans1 = trans1
    self.trans2 = trans2
    self.output_prob = output_prob
    self.initial_prob = initial_prob
    self.synthetic_248 = synthetic_248
    self.num_states = num_states

class ObsSeqs:
  def __init__(self, data, lens, maxlen, num_instances):
    self.data = data
    self.lens = lens
    self.maxlen = maxlen
    self.num_instances = num_instances

class ForwardWrapper:
  def __init__(self):
    self.lib = load_library(f"forward-lib.so")
    self.lib.forward.argtypes = [
      ctypes.c_float,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_float,
      ctypes.c_int64,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int64,
      ctypes.c_int64,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
    ]
    self.lib.forward_merged.argtypes = [
      ctypes.c_float,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_float,
      ctypes.c_int64,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int64,
      ctypes.c_int64,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
    ]

  def forward(self, hmm, seqs, result, alpha1, alpha2):
    self.lib.forward(
      hmm.gamma,
      hmm.trans1.data_ptr(),
      hmm.trans2.data_ptr(),
      hmm.output_prob.data_ptr(),
      hmm.initial_prob.data_ptr(),
      hmm.synthetic_248,
      hmm.num_states,
      seqs.data.data_ptr(),
      seqs.lens.data_ptr(),
      seqs.maxlen,
      seqs.num_instances,
      result.data_ptr(),
      alpha1.data_ptr(),
      alpha2.data_ptr(),
    )

  def forward_merged(self, hmm, seqs, result, alpha1, alpha2):
    self.lib.forward_merged(
      hmm.gamma,
      hmm.trans1.data_ptr(),
      hmm.trans2.data_ptr(),
      hmm.output_prob.data_ptr(),
      hmm.initial_prob.data_ptr(),
      hmm.synthetic_248,
      hmm.num_states,
      seqs.data.data_ptr(),
      seqs.lens.data_ptr(),
      seqs.maxlen,
      seqs.num_instances,
      result.data_ptr(),
      alpha1.data_ptr(),
      alpha2.data_ptr(),
    )
