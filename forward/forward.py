import h5py
import numpy as np
import torch
from math import inf
import triton
import triton.language as tl

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
  hmm = HMM(float(tail_factor), trans1, trans2, out_prob, init_prob, float(synthetic_248), num_states)

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

@triton.jit
def forward_triton_init(
        hmm_initial_prob, hmm_output_prob, hmm_num_states : tl.constexpr, seqs_data,
        seqs_maxlen : tl.constexpr, alpha_src):
  instance = tl.program_id(axis=0)
  o = tl.load(seqs_data + instance * seqs_maxlen)
  state_idx = tl.arange(0, hmm_num_states)
  init_prob = tl.load(hmm_initial_prob + state_idx)
  num_kmers = hmm_num_states // 16
  out_prob = tl.load(hmm_output_prob + o * num_kmers + state_idx % num_kmers)
  tl.store(alpha_src + instance * hmm_num_states + state_idx, init_prob + out_prob)

@triton.jit
def forward_triton_steps(hmm_output_prob, hmm_trans1, hmm_trans2, hmm_gamma : tl.float32, hmm_synthetic_248 : tl.float32, hmm_num_states : tl.constexpr, seqs_data, seqs_lens, seqs_maxlen : tl.constexpr, alpha1, alpha2):
  instance = tl.program_id(axis=0)
  for t in range(1, seqs_maxlen):
    if t & 1:
      alpha_src = alpha1
      alpha_dst = alpha2
    else:
      alpha_src = alpha2
      alpha_dst = alpha1
    state_idx = tl.arange(0, hmm_num_states)
    idx = instance * hmm_num_states + state_idx
    seq_len = tl.load(seqs_lens + instance)
    if t < seq_len:
      o = tl.load(seqs_data + instance * seqs_maxlen + t)
      num_kmers = hmm_num_states // 16

      # Transitively inlined version of forward_prob_predecessors. The loop is
      # unrolled to allow writing to distinct tensors.
      pred = (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p0 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p1 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = 2 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p2 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      pred = 3 * hmm_num_states // 64 + (state_idx // 4) % (hmm_num_states // 64)
      t1 = tl.load(hmm_trans1 + (pred % num_kmers) * 4 + state_idx % 4)
      t2 = tl.load(hmm_trans2 + state_idx // num_kmers)
      p3 = t1 + t2 + tl.load(alpha_src + instance * hmm_num_states + pred)

      # For this part, we have three possible cases, but as Triton reasons about
      # the whole block, we cannot have if-conditions here. We express it using
      # the "tl.where" function, which conditionally chooses values between two
      # tensors based on whether the provided booleans are true or false.

      # if state // 64 == 15
      pred_fst = state_idx
      p_fst = tl.full((hmm_num_states,), hmm_gamma, dtype=tl.float32)

      # else if state // 64 == 14
      pred_snd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
      p_snd = tl.full((hmm_num_states,), hmm_synthetic_248, dtype=tl.float32)

      # else (if state // 64 != 14 && state // 64 != 15)
      pred_trd = ((state_idx // num_kmers) + 1) * num_kmers + state_idx % num_kmers
      p_trd = tl.zeros((hmm_num_states,), dtype=tl.float32)

      # Combination of the three above cases...
      pred = tl.zeros((hmm_num_states,), dtype=tl.int32)
      pred = tl.where(state_idx // num_kmers == 15, pred_fst, pred)
      pred = tl.where(state_idx // num_kmers == 14, pred_snd, pred)
      pred = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, pred_trd, pred)
      p = tl.full((hmm_num_states,), float('-inf'), dtype=tl.float32)
      p = tl.where(state_idx // num_kmers == 15, p_fst, p)
      p = tl.where(state_idx // num_kmers == 14, p_snd, p)
      p = tl.where(state_idx // num_kmers != 14 and state_idx // num_kmers != 15, p_trd, p)
      p4 = p + tl.load(alpha_src + instance * hmm_num_states + pred)

      # Inlined version of log_sum_exp
      maxp = tl.maximum(p0, p1)
      maxp = tl.maximum(maxp, p2)
      maxp = tl.maximum(maxp, p3)
      maxp = tl.maximum(maxp, p4)
      lsexp = maxp + tl.log(tl.exp(p0-maxp) + tl.exp(p1-maxp) + tl.exp(p2-maxp) + tl.exp(p3-maxp) + tl.exp(p4-maxp))
      neginfs = tl.full((hmm_num_states,), float('-inf'), dtype=tl.float32)
      lsexp = tl.maximum(lsexp, neginfs)

      outp = tl.load(hmm_output_prob + o * num_kmers + state_idx % num_kmers)
      tl.store(alpha_dst + idx, lsexp + outp)
    elif seq_len == t:
      alpha_val = tl.load(alpha_src + idx)
      tl.store(alpha_dst + idx, alpha_val)

    # While this barrier is only intended to be use for debugging purposes, it
    # is required here to ensure correct result (Triton v. 2.0.0 - might be
    # fixed in newer versions).
    tl.debug_barrier()

@triton.jit
def forward_triton_lse(hmm_num_states : tl.constexpr, seqs_maxlen : tl.constexpr, alpha1, alpha2, result):
  instance = tl.program_id(axis=0)
  if seqs_maxlen & 1:
    alpha = alpha1
  else:
    alpha = alpha2
  state_idx = tl.arange(0, hmm_num_states)
  alpha_vals = tl.load(alpha + instance * hmm_num_states + state_idx)
  maxp = tl.max(alpha_vals, axis=0)
  psum = tl.sum(tl.exp(alpha_vals - maxp), axis=0)
  tl.store(result + instance, maxp + tl.log(psum))

def forward_triton(hmm, seqs):
  result = torch.empty(seqs.num_instances, dtype=torch.float32, device='cuda')
  alpha1 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  alpha2 = torch.empty(seqs.num_instances * hmm.num_states, dtype=torch.float32, device='cuda')
  grid = lambda meta: (seqs.num_instances, )
  # Skip these cases because it is too slow
  if hmm.num_states > 16384:
    return None
  forward_triton_init[grid](hmm.initial_prob, hmm.output_prob, int(hmm.num_states), seqs.data, int(seqs.maxlen), alpha1)
  forward_triton_steps[grid](hmm.output_prob, hmm.trans1, hmm.trans2, hmm.gamma, hmm.synthetic_248, int(hmm.num_states), seqs.data, seqs.lens, int(seqs.maxlen), alpha1, alpha2)
  forward_triton_lse[grid](int(hmm.num_states), int(seqs.maxlen), alpha1, alpha2, result)

  return result


for kmer_model in ["3mer", "5mer", "7mer"]:
  print(f"Benchmarking {kmer_model} model")

  # Hard-coded paths to particular files for reproducibility.
  model_path = f"data/{kmer_model}-model.hdf5"
  signals_path = f"data/signals.hdf5"
  hmm, seqs = read_trellis_inputs(model_path, signals_path)

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  # Read the expected values given as output from the Trellis model.
  expected = read_expected(f"data/{kmer_model}-expected.txt")

  for i in range(10):
    probs = forward(hmm, seqs)
  start.record()
  for i in range(10):
    probs = forward(hmm, seqs)
  end.record()
  torch.cuda.synchronize()
  print(f"Naive {start.elapsed_time(end) / 10} ms")
  assert torch.allclose(probs, expected), f"{probs}\n{expected}"

  for i in range(10):
    probs = forward_merged(hmm, seqs)
  start.record()
  for i in range(10):
    probs = forward_merged(hmm, seqs)
  end.record()
  torch.cuda.synchronize()
  print(f"Merged {start.elapsed_time(end) / 10} ms")
  assert torch.allclose(probs, expected), f"{probs}\n{expected}"

  for i in range(10):
    probs = forward_triton(hmm, seqs)
  start.record()
  for i in range(10):
    probs = forward_triton(hmm, seqs)
  end.record()
  torch.cuda.synchronize()
  print(f"Triton {start.elapsed_time(end) / 10} ms")
  if probs is not None:
    assert torch.allclose(probs, expected), f"{probs}\n{expected}"
