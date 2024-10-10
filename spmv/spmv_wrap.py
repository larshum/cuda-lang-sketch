import ctypes
import subprocess

def load_library(lib_id):
  try:
    # If the requested library file exists, we assume this is the right one.
    with open(lib_id, "r") as f:
      pass
  except:
    # Compile the generated CUDA code
    cmds = f"nvcc spmv.cu -O3 -lineinfo -arch=native --shared -Xcompiler -fPIC -o {lib_id}".split(' ')
    r = subprocess.run(cmds, capture_output=True)
    if r.returncode != 0:
      print(f"Shared library compilation failed:\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
      exit(r.returncode)
  return ctypes.cdll.LoadLibrary(f"./{lib_id}")

class CSRMatrix:
  def __init__(self, values, rows, cols, nnz, nrows, ncols):
    self.values = values
    self.rows = rows
    self.cols = cols
    self.nnz = nnz
    self.nrows = nrows
    self.ncols = ncols

class SpmvWrapper:
  def __init__(self):
    self.lib = load_library("spmv-lib.so")
    self.lib.spmv.argtypes = [
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_void_p,
      ctypes.c_int64,
      ctypes.c_int64,
      ctypes.c_int64,
      ctypes.c_void_p,
      ctypes.c_void_p,
    ]

  def spmv(self, A, x, y):
    self.lib.spmv(
      A.values.data_ptr(),
      A.rows.data_ptr(),
      A.cols.data_ptr(),
      A.nnz,
      A.nrows,
      A.ncols,
      x.data_ptr(),
      y.data_ptr()
    )
