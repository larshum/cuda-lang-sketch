// Declares the type representing a CSR matrix, including the names and types
// of all of its fields.
type csr_matrix {
  f32 *values;
  i64 *rows;
  i64 *cols;
  i64 nnz;
  i64 nrows;
  i64 ncols;
}

// A task definition, specifying the inputs and outputs as well as the workload
// performed by a particular task.
task spmv_row(csr_matrix A, f32 *x, f32 *y) {
  i64 row = task_index;
  i64 nnz_row = A.rows[row+1] - A.rows[row];
  f32 sum = 0.0;
  parallel for i in A.rows[row] to A.rows[row+1] {
    sum = sum + A.values[i] * x[A.cols[i]];
  }
  y[row] = sum;
}

// This declares a function running on the CPU, which is accessible from the
// outside as a C API.
//
// Preferably, the compiler should also generate a small wrapper library (e.g.,
// in Python) that automatically destructs the 'csr_matrix', so that users do
// not have to pass components of the matrix one by one.
fn spmv(csr_matrix A, f32 *x, f32 *y) {

  // Launches a set of tasks (one task per row of the input matrix). These
  // tasks execute in parallel, independently of each other.
  launch spmv_row[A.nrows](A, x, y);
}
