import ctypes
import subprocess

def load_library(lib_id):
  try:
    # If the requested library file exists, we assume this is the right one.
    with open(lib_id, "r") as f:
      pass
  except:
    # Compile the generated CUDA code
    cmds = f"nvcc reduction.cu -O3 -lineinfo -arch=native --shared -Xcompiler -fPIC -o {lib_id}".split(' ')
    r = subprocess.run(cmds, capture_output=True)
    if r.returncode != 0:
      print(f"Shared library compilation failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
      exit(r.returncode)
  return ctypes.cdll.LoadLibrary(f"./{lib_id}")

class ReductionWrapper:
  def __init__(self):
    self.lib = load_library("reduction-lib.so")
    self.lib.find_max.argtypes = [
      ctypes.c_void_p,
      ctypes.c_int,
      ctypes.c_void_p,
      ctypes.c_void_p,
    ]

  def find_max(self, values, N, max_idx, max_value):
    self.lib.find_max(
      values.data_ptr(),
      N,
      max_idx.data_ptr(),
      max_value.data_ptr()
    )
