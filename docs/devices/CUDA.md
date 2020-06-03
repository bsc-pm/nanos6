# CUDA device tasks

Currently there is support for launching CUDA kernels as tasks, using
CUDA Unified Memory.

The user only needs to provide the kernel code and dependences; device selection,
launch and synchronization are managed by the runtime.

The *device* clause needed for the OmpSs-2 outlined task is `device(cuda)`, and
support needs to be enabled in the runtime using `--with-cuda` configuration flag.

## Writing CUDA tasks

The CUDA kernel code needs to be provided in a `.cu` file, so that it can be compiled by
`nvcc`. We will try to illustrate the procedure using a SAXPY example program.

Example:

```c
#include "cuda-saxpy.hpp"

__global__ void saxpyCUDAKernel(long int n, double a, const double* x, double* y)
{
	long int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}
```

Then the kernel should be annotated as an OmpSs-2 outlined task in the respective header
(in this example `cuda-saxpy.hpp`):

```c
#ifdef __cplusplus
extern "C"
{
#endif

#pragma oss task in([n]x) inout([n]y) device(cuda) ndrange(1, n, 128)
__global__ void saxpyCUDAKernel(long int n, double a, const double* x, double* y);

#ifdef __cplusplus
}
#endif
```

The `ndrange` clause is used to determine the CUDA grid parameters. In this example
`ndrange(1, n, 128)` will result in calling the kernel as:

```c
saxpyCUDAKernel<<<n/128. 128>>>(n, a, x, y);
```

Also, note that in the context of a C++ application, function name mangling has to be disabled
with the use of `extern "C"`.

Then, the kernel can be invoked from another task/part of the program, simply as a function call,
without the need for the explicit CUDA launch. In the example, that could be like:

```c
void saxpy(long int N, long int BS, double a, double *x, double *y) {
	for (long int i = 0; i < N; i += BS) {
		saxpyCUDAKernel(BS, a, &x[i], &y[i]);	/* each call implies a new OmpSs-2 CUDA task */
	}
}
```

As the implementation relies on CUDA Unified Memory, the user should allocate all data that will
be used by CUDA tasks using the appropriate mechanisms:

```c
/* In the main() or other function of a source file, e.g. cuda-saxpy.cpp */
cudaError_t err;
double *x, *y;
err = cudaMallocManaged(&x, N * sizeof(double), cudaMemAttachGlobal);
assert(err == cudaSuccess);
err = cudaMallocManaged(&y, N * sizeof(double), cudaMemAttachGlobal);
assert(err == cudaSuccess);

initialize(N, BS, x, y); /* data initialization procedures */

for (int i = 0; i < ITS; ++i) {
	saxpy(N, BS, a, x, y); /* launch a series of CUDA OmpSs-2 tasks, as shown above */
}
```

## Using cuBLAS in CUDA tasks

cuBLAS calls can be also introduced as CUDA tasks. As the cuBLAS functions need to specify
a CUDA stream handle, which are managed by the runtime in OmpSs-2, Nanos6 provides an API
call to enable the code obtaining the cudaStream the task will run on:

```c
cudaStream_t nanos6_get_current_cuda_stream()
```

This maybe used inside an outlined function, marked with `device(cuda)` in the respective header,
as shown in the example:

```c
void cublasTaskFunction(<args N, x, y etc.>)
{
	cublasSetStream(cublasHandle, nanos6_get_current_cuda_stream());
	cublasDdot(h, N, x, 1, y, 1, result);
}
```

## Compiling OmpSs-2 + CUDA applications

Compilation is derived from the standard OmpSs-2 compilation procedure described in
[README](README.md), using the Mercurium compiler.

To enable the needed transformations for CUDA support, `--cuda` flag needs to be provided to
`mcxx`.

Following the above SAXPY example, compilation can be done by:

```sh
$ mcxx --ompss-2 --cuda cuda-saxpy.cpp saxpy_kernel.cu -o saxpy_app
```

(Note that, for CUDA parts, the C++ backend is needed, therefore always use `mcxx`. For large
applications though, C parts can be compiled with `mcc -c` as described in README).

## Nanos 6 environment variables for CUDA tasks

The runtime provides the following environment variables related to CUDA:

1. `NANOS6_CUDA_STREAMS`: The max number of tasks that can be concurrently run *per device*. Default value is 16.
