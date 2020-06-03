# OpenACC device tasks

Nanos6 supports running OpenACC GPU tasks, using the CUDA Unified Memory
mechanism, provided by the PGI compiler. The user only needs to provide the OpenACC region
code and dependences; device selection, launch and synchronization are managed by the runtime.

The *device* clause needed for the OmpSs-2 outlined task is `device(openacc)` and
support needs to be enabled in the runtime using `--enable-openacc` configuration flag.

For OpenACC device tasks support to work, the system needs to have an installation of
PGI compilers. If the PGI compilers are in the `PATH` variable, detection is automatic by using the
`--enable-openacc` flag. If not in `PATH`, the PGI installation path can be specified
using `--with-pgi=/path/to/pgi/version/`.

**Note**: In both cases (`PATH` or custom folder), it is needed that the *actual* folder
is used and not the *linked* one that the PGI installer usually creates. For example:

A working installation of PGI 19.10 may be in `/opt/pgi/linux86-64-nollvm/19.10`
and the installer has also created `/opt/pgi/linux86-64-nollvm/2019` with links
to the actual installation.

Please use the first one, either you choose to add it in the `PATH` or provide it
as a custom path:
```sh
$ export PATH=/opt/pgi/linux86-64-nollvm/19.10/bin:$PATH
```
or
```sh
$ /path/to/configure --<conf_flags> --enable-openacc --with-pgi=/opt/pgi/linux86-64-nollvm/19.10
```

## Writing OpenACC tasks

As mentioned in general Device-tasks guidelines, OpenACC tasks need to be written
as outlined OmpSs-2 tasks. For instance:

```c
void vector_add(double *A, double *B, double *C, int size)
{
	/* Only absolutely necessary declarations and operations here */

#pragma acc parallel loop
	for (int i = 0; i < size; i++)
		C[i] = A[i] + B[i];

	/* *Do not* put operations here that use *C!!
	 * (or any other variables that are written into)
	 * Taskifying this in OmpSs-2 implies *asynchronous*
	 * launch, therefore appending async(some_queue)
	 * in the acc pragma.

	 * It is possible though to put another
	 * acc kernels or parallel region; the sequence
	 * semantics will be maintained between -async- regions
	 * of the same task, as only one async queue will be used
	 * in the task.

	 * It is, however, suggested to keep regions as separate
	 * tasks in the general case, unless it is actually meaningful
	 * to have more regions in the same task.
	 */
}
```

Then, the outlined task prototype should be declared either in the same
file or in a relevant header:
```c
#pragma oss task device(openacc) in([size]A, [size]B) out([size]C)
void vector_add(double *A, double *B, double *C, int size);
```

### OpenACC features in OmpSs-2 `device(openacc)` tasks

As illustrated in the above example, the scope of using OpenACC in an
OmpSs-2 + OpenACC application differs from a standalone OpenACC program.
OpenACC tasks are provided as a tool, within OmpSs-2 context, to easily offload
compute-heavy, data-parallel regions that can benefit from GPU/Accelerator execution.

To summarize:
 - Use:
	1. Compute constructs (`parallel`, `kernels`, `loop`, `serial` -if you must-),
	together with their respective, execution-related clauses (gangs/workers/vectors,
	collapse, reduction, independent, private/firstprivate etc.)
	1. Atomic: Use `acc atomic` constructs when needed, inside OpenACC regions.
 - Do not use:
	1. `async`, `wait`: OmpSs-2 will do that automatically and compilation will fail if used
	1. Data constructs (`declare`, `data`, `copy`, `copyin/out` etc.): Unified Memory,
	provided by PGI, is used; thus they will be ignored.
	1. Executable directives (`init`, `shutdown`, `update` etc.)
	1. `self` (defined in OpenACC 2.7 and later): It defeats the purpose; use standard OmpSs-2
	SMP tasks for all host-side workload.


Finally, it is possible to have function calls within OpenACC regions. Standard OpenACC
rules apply in this case; if the compiler can't -or isn't allowed to- automatically
inline the function, it must be declared using `#pragma acc routine ...`. as in a
standard OpenACC program.

### OmpSs-2 features in `device(openacc)` tasks

As OpenACC regions are compiled directly to accelerator (CUDA here) code, there are
limitations also to what features of the OmpSs-2 context can be used within OpenACC
tasks' code.

THe following features are not supported
 - Launching OmpSs-2 tasks from OpenACC code.
 - Using `oss atomic`. Please use `acc atomic` (see above, and also *note on atomic*).
 - Nanos6 API calls from OpenACC tasks.

As stated in the previous section, the purpose of OpenACC tasks are to offload
appropriate compute regions. Therefore, the above actions would make little sense
in this context.

### Using allocated data in OpenACC tasks

Use of pointers to user-allocated data is an obvious necessity in probably all programs.
However, our use of PGI-enabled CUDA Unified Memory mechanism for OpenACC dictates some
careful handling, due to implementation-related causes.

In a nutshell:

1. Before calling `malloc` (and friends) or `new`, for data **that will be used in an OpenACC task**
--> insert a `#pragma acc set device_num(0)`. This applies to **all** functions/smp-task regions, and only
needs to be done once per context/region (even if there are multiple allocations in succesion).
1. The code files that contain these allocations **must** be compiled with the PGI-enabled Mercurium (`pgimcc`
or `pgimcxx`, see below in compilation).

<details>
	<summary>Detailed explanation</summary>

## CUDA Unified Memory + multi-threading

For CUDA Unified Memory to work for OpenACC, PGI compilers transform all allocation calls to
`cudaMallocManaged()` (and OpenACC regions to CUDA kernels). In CUDA, for these calls to work,
the thread that is calling them must be associated with a valid *CUDA context*. This is the
default case in an OpenACC program, that is inherently single-threaded from the host side.

In OmpSs-2 context, however, tasks run in different threads (and `main()` is also a task). Therefore,
calling a -PGI-transformed- `cudaMallocManaged()` from a thread (running a task) that has not been associated
with a CUDA context will result in a CUDA Runtime error, or a segmentation fault later on.

Although multi-threaded behaviour is not -yet- defined in OpenACC standard, we inherit this limitation from
the underlying CUDA implementation. The good news is that we also inherit the work-around to it. 
For CUDA + multi-threading it is suggested that each thread does a -dummy- `cudaSetDevice()` operation, that
serves to actually have the CUDA runtime bind the present thread to a CUDA context (see [here](https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/))

If we apply the same principle in OpenACC, the `#pragma acc set device_num(0)` actually translates to `cudaSetDevice(0)`.
As it is a dummy operation, the actual device number doesn't matter; we only use it to achieve binding the
thread-to-context and device 0 is certainly present in all GPU-enabled systems so it is a safe choice.

This work-around has solved all problems we had in this aspect for now, as the OpenACC standard will define these
use-cases in future revisions. It also explains why *all* data that will be used in OpenACC tasks needs to be
in code compiled with PGI, in order to be transformed; other data can obviously be allocated in files compiled with
`mcc`/`mcxx` and will use standard `malloc` (and derivatives).

</details>


## Compiling OmpSs-2 + OpenACC (+ CUDA) applications

As of May 2020, upstream Mercurium has support for OmpSs-2 + OpenACC tasks & PGI compilers in the master/release branch.

Use `--enable-pgi-compilers` when configuring Mercurium and consult Mercurium documentation.

To use OpenACC tasks, it is needed to use PGI-enabled Mercurium, and add the `--openacc` flag:

```sh
$ pgimcc -c --ompss-2 --openacc a_part_in_c.c
$ pgimcxx -c --ompss-2 --openacc a_part_in_cpp.cpp
$ pgimcxx --ompss-2 --openacc a_part_in_c.o a_part_in_c_plus_plus.o -o app
```

Note that the above invocation automatically passes `-fast -acc -ta=tesla:managed` flags to the PGI compilers,
so NVIDIA target devices with unified memory are assumed without need for the user to specify these options.
Additionally, mixed compilation with `mcxx` and `pgimcxx` -used to compile different object files- is also
supported, useful for existing projects that may need to add an OpenACC tasks part on top of the current
code base. However, in these cases the final executable is suggested to be linked with `pgimcxx`.

### Special cases

#### Note on oss atomic

As explained previously, `oss atomic` is not valid in OpenACC tasks; use `acc atomic` pragmas instead.
It is important to note, however, that even if `oss atomic` is used inside a standard, simple SMP task,
the file containing this task **will not** compile with `pgimcc/xx`.

Therefore, if there is a task using `oss atomic` in an OmpSs-2 + OpenACC program:

1. Seperate this task (and, even better, all that are not relevant to OpenACC) in a different source file(s).
1. Compile said file(s) with `mcc`/`mcxx -c file.cpp --ompss-2 --openacc <rest flags>`.
1. Compile *all* OpenACC-related source files (see subsection *Using allocated data* above) with `pgimcc`/`pgimcxx` respectively.
(also see general compiling notes above).

<details>
<summary>Explanation</summary>

As all OmpSs-2 `#pragma` directives, `oss atomic` triggers a source-to-source transformation by Mercurium.

To implement the atomic functionality, Mercurium relies on using specific *GCC* built-in intrinsics (namely `__sync_synchronize()` here).
These intrinsics, although may be supported by other compilers (e.g. IBM), are not supported by PGI. Therefore, when Mercurium
feeds its transformed code to the PGI compiler, compilation fails.

Using plain `mcc/xx` feeds the same code to GCC, as was intended.
</details>

#### Combining with CUDA tasks

It is supported to use both CUDA and OpenACC OmpSs-2 tasks within an application. However, as we have shown,
compilation procedure might be complex, so it is needed to keep things as separated as possible:

 - See [CUDA tasks documentation](docs/devices/CUDA.md) for CUDA compilation details.
 - CUDA tasks *can* take advantage of `malloc`s that PGI transforms to `cudaMallocManaged` as they make use of Unified Memory too,
 thus users can benefit if a CUDA task will be using the same data with an OpenACC task.
 - Hence, CUDA tasks (-->function calls) *can* be called from a file compiled with PGI-Mercurium.
 - CUDA tasks compilation however **must** be done with `mcxx` as shown in the documentation.

## Nanos 6 environment variables for CUDA tasks

The runtime provides the following environment variables related to OpenACC:

1. `NANOS6_OPENACC_DEFAULT_QUEUES`: The number of preallocated async queues *per device*. Default value is 64.
1. `NANOS6_OPENACC_MAX_QUEUES`: The max number of async queues, and therefore tasks, that can be concurrently run *per device*. Default value is 128.

