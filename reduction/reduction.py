import torch

from reduction_wrap import ReductionWrapper

torch.manual_seed(1337)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

N = int(1e6)
x = torch.randn(N, dtype=torch.float32, device='cuda', requires_grad=False)

rw = ReductionWrapper()

start.record()
torch_value, torch_idx = torch.max(x, dim=0)
end.record()
torch.cuda.synchronize()
print(f"Torch {start.elapsed_time(end)} ms")

start.record()
idx = torch.empty(1, dtype=torch.int32, device='cuda', requires_grad=False)
value = torch.empty(1, dtype=torch.float32, device='cuda', requires_grad=False)
rw.find_max(x, N, idx, value)
end.record()
torch.cuda.synchronize()
print(f"CUDA IR {start.elapsed_time(end)} ms")

idx = int(idx)
value = float(value)
assert torch_value == value, f"{torch_value} != {value}"
assert torch_idx == idx, f"{torch_idx} != {idx}"
print("Results OK!")
