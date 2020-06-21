# Nanos6 Runtime

Nanos6 is a runtime that implements the OmpSs-2 parallel programming model,
developed by the [*Programming Models group*](https://pm.bsc.es/)
at the [**Barcelona Supercomputing Center**](http://www.bsc.es/).

## Installation

### Build requirements

To install Nanos6 the following tools and libraries must be installed:

1. automake, autoconf, libtool, pkg-config, make and a C and C++ compiler
1. [boost](http://boost.org) >= 1.59
1. [hwloc](https://www.open-mpi.org/projects/hwloc/)
1. [numactl](http://oss.sgi.com/projects/libnuma/)
1. Finally, it's highly recommended to have a installation of [Mercurium](https://github.com/bsc-pm/mcxx) with OmpSs-2 support enabled. When installing OmpSs-2 for the first time, you can break the chicken and egg dependence between Nanos6 and Mercurium in both sides: on one hand, you can install Nanos6 without specifying a valid installation of Mercurium. On the other hand, you can install Mercurium without a valid installation of Nanos6 using the `--enable-nanos6-bootstrap` configuration flag.

### Optional libraries and tools

In addition to the build requirements, the following libraries and tools enable additional features:

1. [extrae](https://tools.bsc.es/extrae) to generate execution traces for offline performance analysis with [paraver](https://tools.bsc.es/paraver)
1. [elfutils](https://sourceware.org/elfutils/) and [libunwind](http://www.nongnu.org/libunwind) to generate sample-based profiling
1. [graphviz](http://www.graphviz.org/) and pdfjam or pdfjoin from [TeX](http://www.tug.org/texlive/) to generate graphical representations of the dependency graph
1. [parallel](https://www.gnu.org/software/parallel/) to generate the graph representation in parallel
1. [CUDA](https://developer.nvidia.com/cuda-zone) to enable CUDA tasks
1. [PGI](https://pgroup.com) to enable OpenACC tasks
1. [PQOS](https://github.com/intel/intel-cmt-cat) to generate real-time statistics of hardware counters
1. [DLB](https://pm.bsc.es/dlb) to enable dynamic management and sharing of computing resources
1. [jemalloc](https://github.com/jemalloc/jemalloc) to use jemalloc as the default memory allocator, providing better performance than the default glibc implementation. It must be compiled with `--enable-stats` and `--with-jemalloc-prefix=nanos6_je_` to link with the runtime.


## Build procedure

Nanos6 uses the standard GNU automake and libtool toolchain.
When cloning from a repository, the building environment must be prepared through the following command:

```sh
$ autoreconf -f -i -v
```

When the code is distributed through a tarball, it usually does not need that command.

Then execute the following commands:

```sh
$ ./configure --prefix=INSTALLATION_PREFIX ...other options...
$ make all check
$ make install
```

where `INSTALLATION_PREFIX` is the directory into which to install Nanos6.

The configure script accepts the following options:

1. `--with-nanos6-mercurium=prefix` to specify the prefix of the Mercurium installation
1. `--with-boost=prefix` to specify the prefix of the Boost installation
1. `--with-libunwind=prefix` to specify the prefix of the libunwind installation
1. `--with-libnuma=prefix` to specify the prefix of the numactl installation
1. `--with-extrae=prefix` to specify the prefix of the extrae installation
1. `--with-pqos=prefix` to specify the prefix of the PQoS installation
1. `--with-dlb=prefix` to specify the prefix of the DLB installation
1. `--with-jemalloc=prefix` to specify the prefix of the jemalloc installation
1. `--with-cuda[=prefix]` to enable support for CUDA tasks; optionally specify the prefix of the CUDA installation, if needed
1. `--enable-openacc` to enable support for OpenACC tasks. Requires PGI compilers
1. `--with-pgi=prefix` to specify the prefix of the PGI compilers installation, in case they are not in current PATH variable
1. `--enable-monitoring` to enable monitoring and predictions of task/CPU/thread statistics
1. `--enable-chrono-arch` to enable an architecture-based timer for the monitoring infrastructure

The location of elfutils and hwloc is always retrieved through pkg-config.
If they are installed in non-standard locations, pkg-config can be told where to find them through the `PKG_CONFIG_PATH` environment variable.
For instance:

```sh
$ export PKG_CONFIG_PATH=$HOME/installations-mn4/elfutils-0.169/lib/pkgconfig:/apps/HWLOC/2.0.0/INTEL/lib/pkgconfig:$PKG_CONFIG_PATH
```

To enable CUDA. `--with-cuda` flag is needed.
The location of CUDA can be retrieved automatically, if it is in standard system locations (/usr/lib, /usr/include etc.),
or through pkg-config. Alternatively, for non-standard installation paths, it can be specified using the optional `=prefix` of the parameter.

The location of PGI compilers can be retrieved from the PATH variable, if it is not specified through the `--with-pgi` parameter.

After Nanos6 has been installed, it can be used by compiling your C, C++ and Fortran codes with Mercurium using the `--ompss-2` flag.
Example:

```sh
$ mcc -c --ompss-2 a_part_in_c.c
$ mcxx -c --ompss-2 a_part_in_c_plus_plus.cxx
$ mcxx --ompss-2 a_part_in_c.o a_part_in_c_plus_plus.o -o app
```

## Execution

Nanos6 applications can be executed as is.
The number of cores that are used is controlled by running the application through the `taskset` command.
For instance:

```sh
$ taskset -c 0-2,4 ./app
```

would run `app` on cores 0, 1, 2 and 4.

### Scheduling options

The scheduling infrastructure provides the following environment variables to modify the behavior of the task scheduler.

* `NANOS6_SCHEDULING_POLICY=fifo|lifo`: Specifies whether ready tasks are added to the ready queue using a LIFO or a FIFO policy. **fifo** is the default.
* `NANOS6_IMMEDIATE_SUCCESSOR=1|0`: Enables/disables the immediate successor policy. Enabled by default.
* `NANOS6_PRIORITY=1|0`: Enables/disables support for task priorities in the scheduler. Enabled by default.

### Task worksharings options

Worksharing tasks are a special type of tasks that can only be applied to for-loops.
The key point of worksharing tasks is their ability to run concurrently on different threads, similarly to OpenMP parallel fors.
In contrast, worksharing tasks do not force all the threads to collaborate neither introduce any kind of barrier.

An example is shown below:

```c
#pragma oss task for chunksize(1024) inout(array[0;N]) in(a)
for (int i = 0; i < N; ++i) {
    array[i] += a;
}
```

In our implementation, worksharing tasks are executed by taskfor groups.
Taskfor groups are composed by a set of available CPUs.
Each available CPU on the system is assigned to a specific taskfor group.
Then, a worksharing task is assigned to a particular taskfor group, so it can be run by at most as many CPUs (also known as collaborators) as that taskfor group has.
Users can set the number of groups (and so, implicitly, the number of collaborators) by setting the ``NANOS6_TASKFOR_GROUPS`` environment variable.
By default, there are as many groups as NUMA nodes in the system.

Finally, taskfors that do not define any chunksize leverage a chunksize value computed as their total number of iterations divided by the number of collaborators per taskfor group.

## Benchmarking, tracing, debugging and other options

There are several Nanos6 variants, each one focusing on different aspects of parallel executions: performance, debugging, instrumentation, etc.
Nanos6 applications, unlike Nanos++ applications do not require recompiling their code to generate extrae traces or to generate additional information.
This is instead controlled through environment variables, _envar_ from now on, at run time.
Users can select a Nanos6 variant when running an application through the `NANOS6` envar.
The next subsections explain the different variants of Nanos6 and how to enable them.

### Benchmarking

If `NANOS6` envar is not set, the default variant is `optimized`.
This is compiled with high optimization flags, it does not perform validity checks and it does not provide debug information.
That is the variant that should be used when performing benchmarking of parallel applications.

Additionally, Nanos6 offers an extra performant variant named `turbo`, which is the same as `optimized` but adding further optimizations.
Firstly, it enables by default the `discrete` dependency implementation, although users can still change it through the `NANOS6_DEPENDENCIES` envar.
Secondly, it enables two floating-point (FP) unit optimizations in all tasks: flush-to-zero (FZ) and denormals are zero (DAZ).
Please note these FP optimizations could alter the precision of floating-point computations.
In conclusion, enabling all those features can significantly improve the user application's performance out of the box.

### [EXPERIMENTAL] Tracing a Nanos6 application with CTF

To generate a CTF trace, run the application with the `NANOS6` envar set to `ctf`.

A directory named "trace_<binary_name>_<pid>" will be created at the current
working directory at the end of the execution. To visualize this trace, it
needs to be converted to Paraver format first. By default, Nanos6 will convert
the trace automatically at the end of the execution unless the user explicitly
sets the environment variable `NANOS6_CTF2PRV=0`. Please note that the
conversion tool requires python3 and the babeltrace2 package.

It is also possible to conver it manually using the provided converter:

```bash
  ctf2prv <trace>
```

Which will generate the directory <trace>/prv with the Paraver trace.

The ctf2prv tool requires python3 and babeltrace2 python modules. Nanos6 does
not require any package to generate CTF traces.

### Tracing a Nanos6 application with Extrae

To generate an extrae trace, run the application with the `NANOS6` envar set to `extrae`.

Currently there is an incompatibility when generating traces with PAPI.
To solve it, define the following envar:

```sh
$ export NANOS6_EXTRAE_AS_THREADS=1
```

The resulting trace will show the activity of the actual threads instead of the activity at each CPU.
In the future, this problem will be fixed.


### Generating a graphical representation of the dependency graph

To generate the graph, run the application with the `NANOS6` envar set to `graph`.

By default, the graph nodes include the full path of the source code.
To remove the directories, set the `NANOS6_GRAPH_SHORTEN_FILENAMES` envar to `1`.

The resulting file is a PDF that contains several pages.
Each page represents the graph at a given point in time.
Setting the `NANOS6_GRAPH_SHOW_DEAD_DEPENDENCIES` envar to `1` forces future and previous dependencies to be shown with different graphical attributes.

The `NANOS6_GRAPH_DISPLAY` envar, if set to `1`, will make the resulting PDF to be opened automatically.
The default viewer is `xdg-open`, but it can be overridden through the `NANOS6_GRAPH_DISPLAY_COMMAND` envar.

For best results, we suggest to display the PDF with "single page" view, showing a full page and to advance page by page.


### Verbose logging

To enable verbose logging, run the application with the `NANOS6` envar set to `verbose`.

By default it generates a lot of information.
This is controlled by the `NANOS6_VERBOSE` envar, which can contain a comma separated list of areas.
The areas are the following:

<table><tbody><tr><td> <strong>Section</strong> </td><td> <strong>Description</strong>
</td></tr><tr><td> <em>AddTask</em> </td><td> Task creation
</td></tr><tr><td> <em>DependenciesAutomataMessages</em> </td><td> Show messages between automatas with `NANOS6_DEPENDENCIES=discrete`
</td></tr><tr><td> <em>DependenciesByAccess</em> </td><td> Dependencies by their accesses
</td></tr><tr><td> <em>DependenciesByAccessLinks</em> </td><td> Dependencies by the links between the accesses to the same data
</td></tr><tr><td> <em>DependenciesByGroup</em> </td><td> Dependencies by groups of tasks that determine common predecessors and common successors
</td></tr><tr><td> <em>LeaderThread</em> </td><td>
</td></tr><tr><td> <em>TaskExecution</em> </td><td> Task execution
</td></tr><tr><td> <em>TaskStatus</em> </td><td> Task status transitions
</td></tr><tr><td> <em>TaskWait</em> </td><td> Entering and exiting taskwaits
</td></tr><tr><td> <em>ThreadManagement</em> </td><td> Thread creation, activation and suspension
</td></tr><tr><td> <em>UserMutex</em> </td><td> User-side mutexes (critical)
</td></tr></tbody></table>

The case is ignored, and the `all` keyword enables all of them.
Additionally, and area can have the `!` prepended to it to disable it.
For instance, `NANOS6_VERBOSE=AddTask,TaskExecution,TaskWait` is a good starting point.

By default, the output is emitted to standard error, but it can be sent to a file by specifying it through the `NANOS6_VERBOSE_FILE` envar.
Also the `NANOS6_VERBOSE_DUMP_ONLY_ON_EXIT` can be set to `1` to delay the output to the end of the program to avoid getting it mixed with the output of the program.


### Obtaining statistics

To enable collecting timing statistics, run the application with the `NANOS6` envar set to `stats`.

By default, the statistics are emitted standard error when the program ends.
The output can be sent to a file through the `NANOS6_STATS_FILE` envar.

The contents of the output contain the average for each task type and the total task average of the following metrics:

* Number of instances
* Mean instantiation time
* Mean pending time (not ready due to dependencies)
* Mean ready time
* Mean execution time
* Mean blocked time (due to a critical or a taskwait)
* Mean zombie time (finished but not yet destroyed)
* Mean lifetime (time between creation and destruction)

The output also contains information about:

* Number of CPUs
* Total number of threads
* Mean threads per CPU
* Mean tasks per thread
* Mean thread lifetime
* Mean thread running time


Most codes consist of an initialization phase, a calculation phase and final phase for verification or writing the results.
Usually these phases are separated by a taskwait.
The runtime uses the taskwaits at the outermost level to identify phases and will emit individual metrics for each phase.


### Debugging

By default, the runtime is optimized for speed and will assume that the application code is correct.
Hence, it will not perform most validity checks.
To enable validity checks, run the application with the `NANOS6` envar set to `debug`.
This will enable many internal validity checks that may be violated with the application code is incorrect.
In the future we may include a validation mode that will perform extensive application code validation.

To debug an application with a regular debugger, please compile its code with the regular debugging flags and also the `-keep` flag.
This flag will force Mercurium to dump the transformed code in the local file system, so that it will be available for the debugger.

To debug dependencies, it is advised to reduce the problem size so that very few tasks trigger the problem, and then use let the runtime make a graphical representation of the dependency graph as shown previously.

Processing the `NANOS6` envar involves selecting at run time a runtime compiled for the corresponding instrumentation.
This part of the bootstrap is performed by a component of the runtime called "loader.
To debug problems due to the installation, run the application with the `NANOS6_LOADER_VERBOSE` environment variable set to any value.


## Runtime information

Information about the runtime may be obtained by running the application with the `NANOS6_REPORT_PREFIX` envar set, or by invoking the following command:

```sh
$ nanos6-info --runtime-details
Runtime path /opt/nanos6/lib/libnanos6-optimized.so.0.0.0
Runtime Version 2017-11-07 09:26:03 +0100 5cb1900
Runtime Branch master
Runtime Compiler Version g++ (Debian 7.2.0-12) 7.2.1 20171025
Runtime Compiler Flags -DNDEBUG -Wall -Wextra -Wdisabled-optimization -Wshadow -fvisibility=hidden -O3 -flto
Initial CPU List 0-3
NUMA Node 0 CPU List 0-3
Scheduler priority
Dependency Implementation regions (linear-regions-fragmented)
Threading Model pthreads
```

The `NANOS6_REPORT_PREFIX` envar may contain a string that will be prepended to each line.
For instance, it can contain a sequence that starts a comment in the output of the program.
Example:

```sh
$ NANOS6_REPORT_PREFIX="#" ./app
Some application output ...
#	string	version	2017-11-07 09:26:03 +0100 5cb1900		Runtime Version
#	string	branch	master		Runtime Branch
#	string	compiler_version	g++ (Debian 7.2.0-12) 7.2.1 20171025		Runtime Compiler Version
#	string	compiler_flags	-DNDEBUG -Wall -Wextra -Wdisabled-optimization -Wshadow -fvisibility=hidden -O3 -flto		Runtime Compiler Flags
#	string	initial_cpu_list	0-3		Initial CPU List
#	string	numa_node_0_cpu_list	0-3		NUMA Node 0 CPU List
#	string	scheduler	priority		Scheduler
#	string	dependency_implementation	regions (linear-regions-fragmented)		Dependency Implementation
#	string	threading_model	pthreads		Threading Model
```


## Monitoring

Gathering metrics and generating predictions for these metrics is possible and enabled through the Monitoring infrastructure.
Monitoring is an infrastructure composed of several modules.
Each of these modules controls the monitoring and prediction generation of specific elements of the runtime core.
At this moment, Monitoring includes the following modules/predictors:

1. A module to monitor and predict metrics for Tasks
1. A module to monitor and predict metrics for Threads
1. A module to monitor and predict metrics for CPUs
1. A predictor that generates predictions of the elapsed time until completion of the application
1. A predictor that generates real-time workload estimations
1. A predictor that inferrs the CPU usage for a specific time range

All of these metrics and predictions can be obtained at real-time within the runtime.
The infrastructure also includes an external API to poll some of the predictions from user code.
This external API can be used including `nanos6/monitoring.h` in applications.

In addition, checkpointing of predictions is enabled through the Wisdom mechanism.
This mechanism allows saving predictions for later usage, to enable earlier predictions in future executions.

The Monitoring infrastructure is enabled at configure time, however, both the infrastructure and the Wisdom mechanism are controlled through additional environment variables:

* `NANOS6_MONITORING_ENABLE`: To enable/disable monitoring and predictions of task, CPU, and thread statistics. Enabled by default if the runtime is configured with Monitoring.
* `NANOS6_MONITORING_VERBOSE`: To enable/disable the verbose mode for monitoring. Enabled by default if the runtime is configured with Monitoring.
* `NANOS6_MONITORING_ROLLING_WINDOW`: To specify the number of metrics used for accumulators (moving average's window). By default, the latest 20 metrics.
* `NANOS6_WISDOM_ENABLE`: To enable/disable the wisdom mechanism. Disabled by default.

### Known Limitations

Currently, Monitoring capabilities lack support for the `task for` construct.


## Hardware Counters

Nanos6 offers a real-time API to obtain hardware counter statistics of tasks with various backends. The usage of this API is controlled through the `nanos6_hwcounters.json` configuration file, where backends and counters to be monitored are specified. Currently, Nanos6 supports two backends - `papi` and `pqos` - and a subset of their available counters. All the available backends and counters are listed in the default configuration file, found in the scripts folder. To enable any of these, simply modify the `0` in the field and replace it with a `1`.

Next we showcase a simplified version of the configuration file, where the PQoS backend is enabled with a counter that reports the local memory bandwidth and cycles executed of tasks:
```json
{
	"PQOS": {
		"ENABLED": 1,
		"PQOS_MON_EVENT_L3_OCCUP": 0,
		"PQOS_MON_EVENT_LMEM_BW": 1,
		"PQOS_MON_EVENT_RMEM_BW": 0,
		"PQOS_PERF_EVENT_LLC_MISS": 0,
		"PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS": 0,
		"PQOS_PERF_EVENT_UNHALTED_CYCLES": 1
	}
}
```

## Device tasks

For information about using device tasks (e.g., CUDA tasks), refer to the [devices](docs/devices/Devices.md) documentation.

## Cluster support

In order to enable OmpSs-2@Cluster support, you need a working MPI installation in your environment that supports multithreading, i.e. `MPI_THREAD_MULTIPLE`.
Nanos6 needs to be configured with the `--enable-cluster` flag.
For more information, on how to write and run cluster applications see [README-CLUSTER.md](docs/cluster/README-CLUSTER.md).

## Choosing a dependency implementation

The Nanos6 runtime has support for different dependency implementations. The `regions` (or `linear-regions-fragmented`) dependencies are always compiled and are the default implementation. This choice is fully spec-compliant, and supports all of the features. It is also the only implementation that supports OmpSs-2@Cluster and execution workflows.

Other implementations can be compiled in with the corresponding `./configure` flag, and selected dynamically through the `NANOS6_DEPENDENCIES` environment variable.

The available implementations are:

* `NANOS6_DEPENDENCIES=regions`: Supporting all features. **Default** implementation in all Nanos6 variants except for `turbo`.
* `NANOS6_DEPENDENCIES=discrete`: No support for regions nor weak dependencies. Region syntax is supported but will behave as a discrete dependency to the first address, and weaks will behave as normal strong dependencies. Scales better than the default implementation thanks to its simpler logic and is functionally similar to traditional OpenMP model.

## DLB Support

DLB is a library devoted to speed up hybrid parallel applications and maximize the utilization of computational resources. More information about this library can be found [here](https://pm.bsc.es/dlb). To enable DLB support for Nanos6, a working DLB installation must be present in your environment. Configuring Nanos6 with DLB support is done through the `--with-dlb` flag, specifying the root directory of the DLB installation.

After configuring DLB support for Nanos6, its enabling can be controlled at run-time through the `NANOS6_ENABLE_DLB` environment variable. To run with Nanos6 with DLB support then, this variable must be set to true (`export NANOS6_ENABLE_DLB=1`), since by default DLB is disabled.

Once DLB is enabled for Nanos6, OmpSs-2 applications will benefit from dynamic resource sharing automatically. The following example showcases the executions of two applications that share the available CPUs between them:

```sh
# Run the first application using 10 CPUs (0, 1, ..., 9)
taskset -c 0-9   ./merge-sort.test &

# Run the second application using 10 CPUs (10, 11, ..., 19)
taskset -c 10-19 ./cholesky-fact.test &

# At this point the previous applications should be running while sharing resources
# ...
```

## Polling Services

Polling services are executed by a dedicated thread at regular intervals, and also, opportunistically by idle worker threads.
The approximate minimum frequency in time in which the polling services are going to be executed can be controlled by the `NANOS6_POLLING_FREQUENCY` environment variable.
This variable can take an integer value that represents the polling frequency in microseconds.
By default, the runtime system executes the polling services at least every 1000 microseconds.

## CPU Managing Policies

Currently, Nanos6 offers different policies when handlind CPUs through the `NANOS6_CPUMANAGER_POLICY` environment variable:
* `NANOS6_CPUMANAGER_POLICY=idle`: To choose the `idle` policy, in which idle threads halt on a blocking condition, while not consuming CPU cycles.
* `NANOS6_CPUMANAGER_POLICY=busy`: In the `busy` policy, idle threads continue spinning and never halt, consuming CPU cycles.
* `NANOS6_CPUMANAGER_POLICY=lewi`: If DLB is enabled, this policy is the counterpart of the `idle` one, but for the DLB mode. In this policy, idle threads lend their CPU to other runtimes or processes.
* `NANOS6_CPUMANAGER_POLICY=greedy`: If DLB is enabled, the `greedy` policy disables lending CPUs from the process' mask, but allows acquiring and lending external CPUs.
* `NANOS6_CPUMANAGER_POLICY=default`: Fallback to the default implementation. If DLB is disabled, this policy falls back to the `idle` policy, while if DLB is enabled it falls back to the `lewi` policy.

## Throttle

There are some cases where user programs are designed to run for a very long time, instantiating in the order of tens of millions of tasks or more.
These programs can demand a huge amount of memory in small intervals when they rely only on data dependencies to achieve task synchronization.
In these cases, the runtime system could run out of memory when allocating internal structures for task-related information if the number of instantiated tasks is not kept under control.

To prevent this issue, the runtime system offers a `throttle` mechanism that monitors memory usage and stops task creators while there is high memory pressure.
This mechanism does not incur too much overhead because the stopped threads execute other ready tasks (already instantiated) until the memory pressure decreases.
The main idea of this mechanism is to prevent the runtime system from exceeding the memory budget during execution.
Furthermore, the execution time when enabling this feature should be similar to the time in a system with infinite memory.

The throttle mechanism requires a valid installation of Jemalloc, which is a scalable multi-threading memory allocator.
Hence, the runtime system must be configured with the ``--with-jemalloc`` option.
Although the throttle feature is disabled by default, it can be enabled and tunned at runtime through the following environment variables:

* `NANOS6_THROTTLE`: Boolean variable that enables the throttle mechanism. **Disabled** by default.
* `NANOS6_THROTTLE_TASKS`: Maximum absolute number of alive childs that any task can have. It is divided by 10 at each nesting level. By default is 5.000.000.
* `NANOS6_THROTTLE_PRESSURE`: Percentage of memory budget used at which point the number of tasks allowed to exist will be decreased linearly until reaching 1 at 100% memory pressure. By default is 70.
* `NANOS6_THROTTLE_MAX_MEMORY`: Maximum used memory or memory budget. Note that this variable can be set in terms of bytes or in memory units. For example: ``NANOS6_THROTTLE_MAX_MEMORY=50GB``. The default is the half of the available physical memory.
