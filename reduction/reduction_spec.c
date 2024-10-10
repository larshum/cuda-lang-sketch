type MaxType {
  int idx;
  float value;
}

// We specify using annotations that the defined function is both associative
// and commutative, and that we want its definition to be inlined where it is
// used. When 'f' is used in a reduction operation, the compiler consider its
// definition to see whether it has been annotated properly. There are three
// possible cases:
// 1. The function is both associative and commutative. In this case, we use
//    the optimal version of the implementation.
// 2. The function is associative but not commutative. It is still possible to
//    parallelize the reduction in this case, but we have to fall back to a
//    less efficient version.
// 3. Otherwise, we cannot parallelize the reduction. The compiler should
//    report an error (or at least a warning) if the user attempts to reduce
//    with such a function. This would most likely be a mistake on their end.
@associative
@commutative
@inline
def f(MaxType a, MaxType b) : MaxType {
  if a.value > b.value {
    return a;
  } else {
    return b;
  }
}

// Support annotations on arguments passed to functions?
task find_max(float *values, int N, int *max_idx, float *max_value) {
  // Approach using a for-loop to perform a reduction over the function f.
  MaxType acc(0, -1.0 / 0.0);
  reduce f for i in 0 to N {
    acc = f(acc, MaxType(i, values[i]));
  }
  max_idx[0] = acc.idx;
  max_value[0] = acc.value;

  // The main problem with the above reduction loop is that it requires the
  // loop to have a very particular structure that is not enforced
  // syntactically. The compiler has to identify what the initial value is
  // (the value originally assigned to 'acc') and that the second argument to
  // the 'f' function is the expression we map over.
  // * Can we define 'acc' over multiple lines? How would the compiler identify
  //   what part of the code constitutes the initialization?
  // * What if we change the order of arguments passed to the 'f' function?
  //   Would the compiler be able to handle this properly? What if the function
  //   is not commutative?

  // Using a functional style is less volatile because it implicitly enforces a
  // particular structure. However, this requires introducing a higher-order
  // function to compute the value in the i:th iteration of the loop. Below is
  // an alternative sketch of what that might look like.
  MaxType zero(0, -1.0/0.0);
  MaxType max = reduce(f, zero, 0, N, i -> MaxType(i, values[i]));
  max_idx[0] = max.idx;
  max_value[0] = max.value;
}
