# Nanos6 Release Notes
All notable changes to this project will be documented in this file.


## Version 4.1, Wed May 15, 2024
The 4.1 version corresponds to the OmpSs-2 2024.05 release. It introduces the directory/cache (D/C) for Host and CUDA devices. It also adds several fixes for the instrumentation.

### General
- Add directory/cache (D/C) for Host and CUDA devices
- Add device memory allocation API for D/C-managed memory
- Require ovni 1.5.0 or greater
- Fix NUMA tests to accept executions with no `NUMA_ALL_ACTIVE`

### Instrumentation
- Fix thread instrumentation order when blocking task
- Call `ovni_thread_requires` for the Nanos6 model
- Call `ovni_thread_free` when threads end
- Always mark task as paused when entering taskwait in ovni


## Version 4.0, Fri Nov 17, 2023
The 4.0 version corresponds to the OmpSs-2 2023.11 release. It provides support for the [ALPI](https://gitlab.bsc.es/alpi/alpi) tasking interface, reduces the external software requirements, and removes obsolete features.

### General
- Implement the [ALPI](https://gitlab.bsc.es/alpi/alpi) tasking interface to support external task-aware libraries
- Allow embedding jemalloc allocator
- Embed hwloc and jemalloc by default
- Add `devices.cuda.prefetch` option to control CUDA prefetching of data dependencies (enabled by default)
- Install the nanos6.toml configuration file in `$prefix/share`
- Remove obsolete `instrument.h` public interface
- Remove obsolete `stats` and `graph` instrumentations
- Remove software dependency with libunwind and elfutils
- Improve ovni library discovery
- Fix execution when enabling extrae instrumentation
- Fix memory leaks
- Fix several tests


## Version 3.0, Wed May 24, 2023
The 3.0 release corresponds to the OmpSs-2 2023.05 release. It introduces several performance improvements, important bug fixes, and improved usability and programmability. It also improves the support for the ovni instrumentation.

### General
- Leverage C++17 standard, which may require newer GCC (such as GCC 7 or later)
- Fix visualization of task labels for programmatically spawned tasks (e.g., polling tasks from task-aware libraries)
- Deprecate CTF instrumentation; use ovni instrumentation instead
- Remove support for the `task for` clause

### Performance
- Decrease the default immediate successor probability to 0.75 instead of 1.0. Always applying the immediate successor policy can degrade the performance of some applications. Instead, if your program considerability relies on it, set it back to 1.0
- Remove several dynamic memory allocations that were on the critical path of Nanos6 code
- The `turbo.warmup` indicates whether the runtime should perform a warmup of Jemalloc arenas (enabled by default)
- Add the config list option `cpumanager.sponge_cpus` to indicate which CPUs should not be used by the runtime. A sponge CPU is a CPU that the runtime system has available but it does not execute any task (or runtime code) on it. Such CPUs are useful to reduce the system noise. The runtime leaves these CPUs free (without consuming CPU time) so that the system can schedule other threads and interruptions on them

### Building and Usability
- Add the `autogen.sh` script to prepare autotools, instead of `autoreconf`
- Allow embedding a hwloc library into Nanos6 to avoid conflicts with other external hwloc libraries
- Add the configure option `--with-hwloc` to specify whether hwloc should be external or embedded
- Attach the hwloc 2.9.1 tarball inside the `deps` folder for the embedded default hwloc. See `autogen.sh --help` for more information

### **ovni** Instrumentation
- Add support for the ovni's Idle view that can be displayed with Paraver
- Add support for the ovni's Breakdown view that can be displayed with Paraver
- Support ovni 1.2.0 version and higher compatible versions
- Perform a run-time version check to verify if the loaded ovni library is compatible
- Link Nanos6 with ovni library using `runpath` instead of `rpath` to allow changing the ovni library through `LD_LIBRARY_PATH`


## Version 2.8, Tue Nov 15, 2022
The 2.8 release corresponds to the OmpSs-2 2022.11 release. It introduces LLVM support for CUDA tasks and runtime loading. It also adds OVNI instrumentation support and several bug fixes and features that improve the overall performance and programmability.

### General
- Add a probabilistic attribute to the Immediate Successor feature (enabled by default)
- Improve locking, reduce allocations, and fix alignment issues
- Improve task creation performance

### **ovni** Instrumentation
- Add support for **ovni** instrumentation through the configuration: `version.instrument=ovni`
- Choose detail level with the option `instrument.ovni.level`

### LLVM Support for CUDA
- Support `device(cuda)` tasks in OmpSs-2 programs built with the LLVM compiler
- Drop support for `device(cuda)` in Mercurium
- Support both building kernels separately with NVCC and linking them to the final binary or building directly with LLVM
- Support building PTX binaries and CUDA kernels at runtime when placed in a specific folder (by default nanos6-cuda-kernels)
- Add a new configuration option for the default CUDA kernels folder `devices.cuda.kernels_folder`


## Version 2.7.1, Fri May 27, 2022
The 2.7.1 release corresponds to the OmpSs-2 2021.11.1 release. It introduces some bug and code fixes,
and some minor improvements

### General
- Adapt taskfor to avoid overwriting task args in compiler-generated code
- Improve support for custom CXXFLAGS at configure time
- Add `--disable-all-instrumentations` configure option
- Modify API to allow setting task labels
- Provide `nanos6-info` with new options to show compile/link runtime flags
- Provide `nanos6-info` with new options to show current and default config files


## Version 2.7, Wed Nov 17, 2021
The 2.7 release corresponds to the OmpSs-2 2021.11 release. It introduces some performance and code fixes,
and several fixes for the CTF tracing tools.

### General
- Set `hybrid` CPU manager policy as default
- Fix the setting of a floating-point optimization bit in the CSR register (x86) when enabling `turbo` mode
- Add several fixes to CTF tracing tools
- Add support for `if(0)` and taskwaits with dependencies in fast CTF converter (`nanos6-ctf2prv-fast`)
- Remove unnecessary warning at run-time in the NUMA-aware code


## Version 2.6, Wed Jun 30, 2021
The 2.6 release corresponds to the OmpSs-2 2021.06 release. It introduces several features and fixes that improve
the general performance and programmability. It introduces a NUMA-aware API to allocate structures in taskified
applications. It also adds a new and faster CTF trace converter and support for multi-process tracing.

### General
- Add NUMA-aware API to allocate structures in taskified programs
- Add new hybrid CPU manager policy that combines both idle and busy policies
- Make idle CPUs block inside the scheduler while there are no ready tasks when enabling the busy policy
- Add support for the `onready` task clause
- Consolidate the new polling mechanism using the `nanos6_wait_for` function
- Remove polling services API
- Remove OmpSs-2@Cluster features and code

### Instrumentation
- Add fast CTF trace converter enabled through the config option `instrument.ctf.converter.fast`
- Add support for multi-process tracing enabled by the Task-Aware MPI and Task-Aware GASPI libraries
- Add merger tool for multi-process traces named `nanos6-mergeprv`


## Version 2.5.1, Tue Dec 22, 2020
The 2.5.1 release corresponds to the OmpSs-2 2020.11.1 release. It introduces bug fixes and code improvements.

### General
- Unify instrumentation, monitoring and hwcounter points
- Efficient support for taskloop dependencies
- Fix reductions in taskloops and taskfors
- Centralize configuration variables
- Fully implement `assert` directive
- Abort execution when an invalid config variable is enabled
- Fix CTF instrumentation bugs
- Bugfixes, performance and code improvements


## Version 2.5, Wed Nov 18, 2020
The 2.5 release corresponds to the OmpSs-2 2020.11 release. It introduces several features and fixes that improve
general performance. It replaces the configuration environment variables with a configuration file, improving the
usability of the runtime system. It also makes the discrete dependency system the default implementation.

### General
- Replace all environment variables with a configuration file
- Add `NANOS6_CONFIG` environment variable to specify the configuration file
- Add `NANOS6_CONFIG_OVERRIDE` to override options from the configuration file
- Enhance performance in architectures with hyperthreading
- Improve locking performance
- Allocate critical C++ containers with the custom memory allocator
- Support the `assert` directive to check the loaded dependency system

### Dependency System
- Make `discrete` the default dependency system
- Improve allocations in discrete dependencies
- Add support for CUDA task reductions in discrete dependencies
- Use address translation tables for specifying task reductions' storage

### Instrumentation
- Add support for kernel events in CTF instrumentation
- Add new Paraver views for CTF traces

### Devices
- Add fixes for OpenACC and CUDA devices


## Version 2.4.1, Tue Sep 22, 2020
The 2.4.1 release corresponds to the OmpSs-2 2020.06.1 release. It introduces bug fixes and performance improvements.

### General
- Improve the interface and performance of the scheduler's lock
- Fix CTF instrumentation bugs and limitations
- Fix PAPI hardware counters backend
- Support newer versions of GCC, Clang and GLIBC
- Fix task external events API
- Remove preemption mechanism from critical sections
- Fix initialization of locks
- Add test suite built with the OmpSs-2 compiler based on LLVM
- Add new tests


## Version 2.4, Mon Jun 22, 2020
The 2.4 release corresponds to the OmpSs-2 2020.06 release. It introduces several features that improve the general
performance of OmpSs-2 applications. It adds a new variant to extract execution traces with a lightweight internal
tracer. It also improves the support for CUDA and provides support for OpenACC tasks.

### General
- Use jemalloc as a scalable multi-threading memory allocator
- Add `turbo` variant enabling floating-point optimizations and the discrete dependency system
- Refactor of CPU Manager and DLB support improvements
- Bugfixes, performance and code improvements

### Scheduling
- Improve taskfor distribution policy
- Improve scheduling performance and code
- Add the `nanos6_wait_for` function to efficiently pause a task for a given time

### Dependency System
- Implement the discrete dependency system with lock-free techniques
- Add support for weak dependencies in discrete
- Add support for commutative and concurrent dependencies in discrete

### Instrumentation
- Refactor the hardware counters infrastructure and support both PAPI and PQoS counters
- Add `ctf` variant to extract execution traces in CTF format using a lightweight internal tracer
- Provide the `ctf2prv` tool to convert CTF traces to Paraver traces
- Avoid Extrae trace desynchronizations in hybrid MPI+OmpSs-2 executions
- Remove the `stats-papi` instrumentation variant

### Devices
- Refactor of the devices' infrastructure
- Perform transparent CUDA Unified Memory prefetching
- Add support for cuBLAS and similar CUDA APIs
- Add support for OpenACC tasks


## Version 2.3.2, Wed Jan 8, 2020
The 2.3.2 release corresponds to the OmpSs-2 2019.11.2 release. It mainly introduces bug fixes.

### General
- Fix important error at the runtime initialization
- Fix in discrete dependency system
- Several fixes for OmpSs-2@Cluster


## Version 2.3.1, Tue Dec 10, 2019
The 2.3.1 release corresponds to the OmpSs-2 2019.11.1 release. It introduces bug fixes and performance improvements.

### General
- Fix execution of CUDA tasks
- Fix `dmalloc` in OmpSs-2@Cluster
- Add missing calls to CPU Manager
- Improve taskfor performance
- Improve general performance by using a reasonable cache line size padding
- Add tests checking the execution of CUDA tasks


## Version 2.3, Mon Nov 18, 2019
The 2.3 release corresponds to the OmpSs-2 2019.11 release. It introduces a new optimized data dependency implementation.
It improves the usability, performance and code of the scheduling infrastructure and the `task for` feature. It also adds
support for DLB and OmpSs-2@Linter.

### General
- Data dependency implementation can be decided at run-time through `NANOS6_DEPENDENCIES` variable
- Performance and code improvements on the `task for` feature
- Add support for Dynamic Load Balancing (DLB) tool
- Add support for [OmpSs-2@Linter](https://github.com/bsc-pm/ompss-2-linter)
- Important bugfix in memory allocator (used by OmpSs-2@Cluster)
- Bugfixes, performance and code improvements

### Dependency System
- Add new optimized discrete dependency system implementation; enabled by `NANOS6_DEPENDENCIES=discrete`

### Scheduling
- Usability, performance and code improvements on the scheduling infrastructure

### Instrumentation
- Remove profile instrumentation variant
- Remove interception mechanism of memory allocation functions


## Version 2.2.2, Mon Oct 7, 2019
The 2.2.2 release corresponds to the OmpSs-2 2019.06.2 release. It introduces bug fixes.

### General
- Compile extrae variant with high optimization flags
- Remove backtrace sampling from the extrae variant


## Version 2.2.1, Fri Sep 27, 2019
The 2.2.1 release corresponds to the OmpSs-2 2019.06.1 release. It mainly introduces bug fixes and code improvements.

### General
- Rename loop directive to task for
- Tasks can leverage reductions and external events at the same time (over distinct data regions)
- OmpSs-2@Cluster bugfixes
- Fix binding information reported by nanos6-info binary
- Support for the TAGASPI library
- Other bugfixes and code improvements


## Version 2.2, Mon Jun 17, 2019
The 2.2 release corresponds to the OmpSs-2 2019.06 release. It mainly introduces the new support for OmpSs-2@Cluster. It also
includes some improvements and optimizations for array task reductions and general bugfixes.

### General
- Support for OmpSs-2@Cluster
- Bugfixes and performance improvements

### Dependency System
- Bugfixes and optimization for array reductions
- Delete obsolete task data dependency implementations

### Scheduling
- Delete obsolete schedulers


## Version 2.1, Fri Nov 9, 2018
The 2.1 release corresponds to the OmpSs-2 2018.11 release. It provides full support for the [TAMPI](https://github.com/bsc-pm/tampi)
library. It also includes general bugfixes and performance improvements.

### General
- Full support for TAMPI
- Bugfixes and performance improvements

### Others
- Bugfixes in task external events API


## Version 2.0.2, Mon Jun 25, 2018
The 2.0.2 release corresponds to the OmpSs-2 2018.06.2 release.

### General
- Bugfixes in HWLOC support


## Version 2.0.1, Mon Jun 25, 2018
The 2.0.1 release corresponds to the OmpSs-2 2018.06.1 release.

### General
- Bugfixes in task reductions


## Version 2.0, Mon Jun 25, 2018
The 2.0 release corresponds to the OmpSs-2 2018.06 release. It introduces support for OmpSs-2@CUDA in Unified Memory NVIDIA devices.
It also supports array task reductions in C/C++ and task priorities. Additionally, it provides two new APIs used by the
[TAMPI](https://github.com/bsc-pm/tampi) library.

### General
- Support for OmpSs-2@CUDA Unified Memory
- Bugfixes and performance improvements

### Dependency System
- Support for array task reductions in C/C++

### Scheduling
- Support for task priorities
- Add priority scheduler

### Others
- Add polling services API
- Add task external events API
- Rename taskloop construct to loop


## Version 1.0.1, Thu Nov 23, 2017
The 1.0.1 release corresponds to the OmpSs-2 2017.11.1 release.

### General
- Fixes for the building system
- Fixes for the loading system


## Version 1.0, Mon Nov 13, 2017
The 1.0 release corresponds to the OmpSs-2 2017.11 release. It is the first release of the Nanos6 runtime system. It implements the basic
infrastructure to manage the parallelism of user tasks (task creation, task scheduling, etc) and their data dependencies. The task dependency
system supports the nested dependency domain connection, and both early release and weak dependency models.

### General
- General infrastructure of the runtime system
- Support for user tasks and nesting of tasks

### Scheduling
- Implement different schedulers: FIFO, LIFO, etc

### Dependency System
- Implementation of a task data dependency system
- Support for nested dependency domain connection
- Support for early release of task dependencies
- Support for weak task dependencies
- Support for reductions

### Others
- Taskloop construct with dependencies
- Task pause/resume API
