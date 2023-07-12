# CUDA device tasks

Currently, there is support for launching CUDA kernels as tasks relying on the
CUDA Unified Shared Memory (USM). Such tasks can run on the GPUs and access
data allocated as CUDA managed memory, which is trasparently accessible from
the host and the devices.

To instantiate kernel functions as device tasks, the user has to annotate the
kernel functions as device tasks and express their data dependencies as regular
tasks. The user provides the kernel code and indicates the thread hierarchy
that will be used to launch the kernel. Then, the runtime system automatically
manages both the launching and synchronization of these kernel tasks.

A device task must be declared always as an outline task and must define the
`device` clause to the corresponding device type. For instance, in the case of
CUDA tasks, the clause must be `device(cuda)`. Remember that the runtime system
must be configured and compiled with CUDA support using the `--with-cuda`
configure option.

The CUDA device tasks also need to specify the thread hierarchy used to execute
the kernel in the GPU using the `ndrange` clause. The clause has the format:

```c
ndrange(n, G1, ..., Gn, L1, ..., Ln)
```

where the `n` parameter determines the number of dimensions, (i.e., 1, 2 or 3),
and the `Gx` and `Lx` are the sequence of scalars determining the global and
local sizes, respectively. There will be as many `G` and `L` as the number of
dimensions.

The global sizes are the total number of elements to be processed by the kernel
per dimension. These are the total number of threads that will be spawn in each
dimension. Notice that this does not correspond to the CUDA grid sizes. Then,
the local sizes are the number of threads per local block in each dimension. In
this case, the local sizes correspond to the CUDA block sizes. For instance,
the clause `ndrange(2, 1024, 1024, 128, 128)` will create a device task with a
thread hierarchy of 1024x1024 elements, grouped in blocks of 128x128 elements.

Finally, the `shmem` clause allows specifying the amount of dynamic CUDA shared
memory bytes that a kernel task will leverage. Shared memory is fast access
on-chip memory that is allocated per thread block; the same memory is accessible
by all the threads of the same block. The runtime allocates this amount of bytes
when launching a kernel. The shared memory should be handled by the user in the
kernel code. If the clause is not specified, the runtime does not allocate shared
memory.

## Writing CUDA device tasks

The CUDA kernel code needs to be provided in a `.cu` file, so that it can be
compiled by LLVM/Clang and treated as CUDA code. We illustrate the complete
procedure using a SAXPY example program. The kernel is implemented in the
`cuda-saxpy-kernel.cu` file:

```c
#include <cuda_runtime.h>

#include "cuda-saxpy.hpp"

__global__ void saxpyCUDAKernel(long int n, double a, const double* x, double* y)
{
	long int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}
```

Then, the kernel should be annotated as an OmpSs-2 outline task in the respective
header, which is `cuda-saxpy.hpp` in this case. This header must be included from
any source that needs to instantiate that device task. The header is shown below:

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

The `ndrange` clause is used to determine the thread hierarchy of the kernel. In
this example, `ndrange(1, n, 128)` will result in calling the kernel as:

```c
saxpyCUDAKernel<<<n/128. 128>>>(n, a, x, y);
```

Notice that you may need to declare the kernel as `extern "C"` to guarantee C
linkage and avoid function mangling. This may be necessary in applications that
have C source files.

Now, the device task can be instantiated from another part of the program by
simply calling the function, as any regular outline task. There is no need to
call any CUDA function. In the SAXPY example, a CUDA task can be instantiated
to process one data block of `BS` elements as follows:

```c
void saxpy(long int N, long int BS, double a, double *x, double *y)
{
	for (long int i = 0; i < N; i += BS) {
		// Each call implies a new OmpSs-2 CUDA task and will transparently
		// launch the corresponding CUDA kernel into the GPU
		saxpyCUDAKernel(BS, a, &x[i], &y[i]);
	}
	#pragma oss taskwait
}
```

As the implementation relies on CUDA Unified Shared Memory, the user should
allocate all data that will be used by CUDA tasks using the appropriate CUDA
mechanisms. The data must be allocated through `cudaMallocManaged` and passing
the `cudaMemAttachGlobal` flag to obtain USM memory. In the SAXPY example, the
`main` function, defined in `cuda-saxpy.cpp`, allocates managed memory for the
`x` and `y` arrays:

```c
#include <cuda_runtime.h>

#include "cuda-saxpy.hpp"

int main()
{
	long int N = ...;

	double *x, *y;
	cudaMallocManaged(&x, N * sizeof(double), cudaMemAttachGlobal);
	cudaMallocManaged(&y, N * sizeof(double), cudaMemAttachGlobal);

	initialize(N, BS, x, y);

	saxpy(N, BS, a, x, y);

	cudaFree(x);
	cudaFree(y);
}
```

## Using cuBLAS and other CUDA math libraries in CUDA tasks

The cuBLAS operations can be also introduced as CUDA tasks. As the cuBLAS functions
need to specify a CUDA stream handle, which are managed by the runtime in OmpSs-2,
Nanos6 provides an API function to obtain the stream associated to the device task:

```c
cudaStream_t nanos6_get_current_cuda_stream(void);
```

Notice that such device tasks launch kernels to the device but they do it from
the host side. Thus, these are a special kind of device task which run on the
host. Such special device tasks must be marked with `device(cuda)` but without
the `ndrange` clause. In this case, these tasks can be declared outline or inline,
in contrast to the pure device tasks. Altough they run directly on the host, they
are associated to a specific CUDA stream, as mentioned above.

As an example:

```c
#pragma oss task device(cuda)
void cublasTaskFunction(cublasHandle_t handle, int N, const double *x, const double *y, double *res)
{
	cublasSetStream(handle, nanos6_get_current_cuda_stream());
	cublasDdot(handle, N, x, 1, y, 1, res);
}
```

Please note that a single cuBLAS context cannot be reused for multiple devices.
This means that the above example will not be enough when running in a system
with multiple GPUs, because Nanos6 can schedule tasks in any CUDA device. A
possible way to tackle this problem is to create one cuBLAS context for each
GPU and then select inside the task the appropiate handle using the
`cudaGetDevice` call to get the GPU the task will be running on.

## Compiling OmpSs-2@CUDA applications

Compilation is derived from the standard OmpSs-2 compilation procedure described
in [README](README.md), using the LLVM/Clang compiler. Following the above SAXPY
example, compilation can be done by:

```sh
$ clang++ -fompss-2 cuda-saxpy.cpp cuda-saxpy-kernel.cu -o saxpy -lcudart
```

Notice the application must be linked to the CUDA runtime library (`libcudart`)
since it calls functions of the CUDA runtime library. LLVM/Clang will try to use
the default CUDA installation on your system. To specify a different CUDA toolchain
or get other information on compiling CUDA with LLVM/Clang, please check the
[LLVM documentation](https://llvm.org/docs/CompileCudaWithLLVM.html).

## Runtime options for CUDA tasks

The runtime provides the following configuration variables related to CUDA:

1. `devices.cuda.streams`: The maximum number of tasks that can be concurrently
run *per device*. Default value is 16.
1. `devices.cuda.page_size`: The CUDA device page size. Default value is 0x8000.
1. `devices.cuda.polling.pinned`: Indicates whether the CUDA polling services
should constantly run while there are CUDA tasks running on their GPU. Enabling
this option may reduce the latency of processing CUDA tasks at the expenses of
occupiying a CPU from the system. Default value is true.
1. `devices.cuda.polling.period_us`: The time period in microseconds between CUDA
service runs. During that time, the CPUs occupied by the services are available
to execute ready tasks. Setting this option to 0 makes the services to constantly
run. Default value is 1000 microseconds.
