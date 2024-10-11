# cuda-lang-sketch

This repository contains a few examples illustrating the main idea of the IR and what the resulting code might look like.

## Structure

Each directory `D` containing a complete example consists of the following four files:
* `<D>_spec.c` represents the IR file (the file extension is TBD).
* `<D>.cu` is a sketch of the output CUDA code from the IR compiler.
* `<D>_wrap.py` is a sketch of a Python wrapper output from the IR compiler, meant to simplify use of the generated code (when the IR code is self-contained, and does not refer to external code we need to link it with).
* `<D>.py` is the main runner that benchmarks the "generated" code somehow.

## Task Idea

The main idea of this IR that differentiates it from other approaches is the task style of parallelism. Each task represents an independent workload which may execute sequentially or in parallel (determined by user annotations/uses of special keywords).

We can launch a set of independent tasks in parallel as `launch taskid[nparallel](args...)`, and a series of launches are executed in sequence (the first must finish before the second one starts, as is the case for CUDA kernels). This approach should work well for most cases. The compiler can make (at least) four decisions on how to map a task launch to a CUDA kernel launch:
1. If the task is inherently sequential (or we have too little parallelism), each task is mapped to a single GPU thread. In this case, each block runs multiple tasks.
2. If requested by the user, we could run one task per warp (i.e., by having the other 31 threads of each warp idle). This could be useful in certain cases, for instance when we have too much control flow to benefit from running it in parallel, but it is important to keep data on the GPU.
3. When the task has a lot of parallelism available, each task is mapped to one or more warps. Typically, we would have one block of warps per task (but this could be configurable).
4. If the user wants to, each task could be mapped to multiple blocks. This is useful when we have a lot of available parallelism and no need to synchronize. If we need synchronization within the task (e.g., when we do a reduction), this would require splitting up the task into multiple kernels (or use the cooperative groups API, if that is as efficient).

For certain problems (e.g., ray tracing), we may want to define tasks that launch other tasks. I have not made a sketch of what this might look like, but generally, we would have two options:
1. Launch a set of independent tasks in parallel as above. In this case, a recursive task launch becomes a recursive kernel launch (by relying on the CUDA dynamic parallelism).
2. Launch the tasks using a task queue. In this case, any nested launches result in a push to the queue, while a thread/warp/block (depending on how tasks are mapped to the GPU) reads from the queue when they are idle, and this keeps going until the queue is empty. Users would be responsible for configuring the queue (e.g., max size) via annotations.

## Related Work

The main benefit of the task model is that the user only needs to reason about one dimension of parallelism (how many tasks to launch in parallel). In CUDA, we both need to decide how many blocks to launch and the number of threads per block (and we can also have multi-dimensional launches). We can still achieve the same expressivity by allowing users to annotate how the launch of a task set should be performed (e.g., specify that each task should be mapped to 32 threads).

Compared to Triton, the task approach allows us to more efficiently handle problems with irregular parallelism. For instance, consider the `spmv` example, where each task (our IR) or program (Triton) handles a full row. In our IR, we can express this as a single loop (annotated to indicate it should run in parallel) summing the products, across all non-zeros of the row. In Triton, however, we cannot write this kind of code, because it requires a statically known number of non-zeros. We work around it by summing over fixed blocks of the row at a time. This is less efficient, because it results in a block-level reduction per iteration, whereas our (hypothetical) code only needs to do it once at the end.

The division of work into tasks is similar to how it is done in Legion/Regent, but the aim is rather different (as their focus is on distributed workloads). First, our tasks are more fine-grained, and can map to individual threads on the GPU, whereas Legion tasks are mapped to an entire processor (in my understanding). Second, Legion schedules tasks at runtime, whereas our standard launch approach is static - this is important, because the extra runtime overhead due to this matters a lot for us.
