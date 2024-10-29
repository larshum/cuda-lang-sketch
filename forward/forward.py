import h5py
import numpy as np
import torch
from math import inf

from forward_wrap import HMM, ObsSeqs, ForwardWrapper

def generate_init_probs(k):
  init_probs = np.zeros((16, 4**k), dtype=np.float32)
  for kmer in range(0, 4**k):
    init_probs[0][kmer] = np.log(1.0 / float(4**k))
  for layer in range(1, 16):
    for kmer in range(0, 4**k):
      init_probs[layer][kmer] = -inf
  return init_probs

def reverse_index(i, k):
  return sum([(i // 4**x) % 4 * (4**(k-x-1)) for x in range(k)])

def transform_output_probs(obs, k):
  output_probs = np.zeros((4**k, 101), dtype=np.float32)
  for i in range(4**k):
    idx = reverse_index(i, k)
    for j in range(101):
      output_probs[i][j] = obs[j][idx]
  return output_probs.transpose()

def read_trellis_inputs(model_path, signals_path):
  with h5py.File(model_path, "r") as f:
    with np.errstate(divide="ignore"):
      obs = np.log(f['Tables']['ObservationProbabilities'][:])
    trans1 = np.log(f['Tables']['TransitionProbabilities'][:])
    duration = np.log(f['Tables']['DurationProbabilities'][:])
    tail_factor = np.log(f['Tables']['DurationProbabilities'].attrs['TailFactor'])
    k = f['Parameters'].attrs['KMerLength']
    init_probs = generate_init_probs(k)
    trans1 = trans1.reshape(4, 4**k).transpose(1, 0).flatten()
    out_prob = transform_output_probs(obs, k).flatten()
  with h5py.File(signals_path, "r") as f:
    keys = list(f.keys())
    signals = [f[k]['Raw']['Signal'][:].tolist() for k in keys]
  synthetic_248 = np.log(np.exp(0.) - np.exp(tail_factor))
  num_states = 16 * 4**k

  # Convert data to torch compatible format allocated on the GPU.
  trans1 = torch.tensor(trans1, dtype=torch.float32, device='cuda')
  trans2 = torch.tensor(duration, dtype=torch.float32, device='cuda')
  out_prob = torch.tensor(out_prob, dtype=torch.float32, device='cuda')
  init_prob = torch.tensor(init_probs, dtype=torch.float32, device='cuda')
  hmm = HMM(tail_factor, trans1, trans2, out_prob, init_prob, synthetic_248, num_states)

  signal_lengths = [len(s) for s in signals]
  maxlen = max(signal_lengths)
  torch_signals = torch.empty(maxlen * len(signals), dtype=torch.uint8, device='cuda')
  for i, s in enumerate(signals):
    ofs = i * maxlen
    torch_signals[ofs:ofs+len(s)] = torch.tensor(s, dtype=torch.uint8, device='cuda')
  lens = torch.tensor(signal_lengths, dtype=torch.int64, device='cuda')
  num_instances = len(lens)
  seqs = ObsSeqs(torch_signals, lens, maxlen, num_instances)
  return hmm, seqs

def read_expected(fname):
  with open(fname) as f:
    return torch.tensor([float(l) for l in f.readlines()], dtype=torch.float32, device='cuda')

fw = ForwardWrapper()
def forward(hmm, seqs):
  result = torch.empty(seqs.num_instances, dtype=torch.float32, device='cuda')
  alpha1 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  alpha2 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  fw.forward(hmm, seqs, result, alpha1, alpha2)
  return result
def forward_merged(hmm, seqs):
  result = torch.empty(seqs.num_instances, dtype=torch.float32, device='cuda')
  alpha1 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  alpha2 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  fw.forward_merged(hmm, seqs, result, alpha1, alpha2)
  return result

# Hard-coded paths to particular files for reproducibility.
model_path = "data/model.hdf5"
signals_path = "data/signals.hdf5"
hmm, seqs = read_trellis_inputs(model_path, signals_path)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Read the expected values given as output from the Trellis model.
expected = read_expected("data/expected.txt")

for i in range(10):
  probs = forward(hmm, seqs)
start.record()
for i in range(10):
  probs = forward(hmm, seqs)
end.record()
torch.cuda.synchronize()
print(f"Naive {start.elapsed_time(end)} ms")

assert torch.allclose(probs, expected), f"{probs}\n{expected}"

for i in range(10):
  probs = forward_merged(hmm, seqs)
start.record()
for i in range(10):
  probs = forward_merged(hmm, seqs)
end.record()
torch.cuda.synchronize()
print(f"Merged {start.elapsed_time(end)} ms")

assert torch.allclose(probs, expected), f"{probs}\n{expected}"
