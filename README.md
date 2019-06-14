# Nanos6 Runtime

Nanos6 is a runtime that implements the OmpSs-2 parallel programming model,
developed by the [*Programming Models group*](https://pm.bsc.es/)
at the [**Barcelona Supercomputing Center**](http://www.bsc.es/).

## Installation

### Build requirements

To install Nanos6 the following tools and libraries must be installed:

1. automake, autoconf, libtool, make and a C and C++ compiler
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
1. [PAPI](https://icl.cs.utk.edu/papi/software/index.html)  to generate statistics that include hardware counters
1. [CUDA](https://developer.nvidia.com/cuda-zone) to enable CUDA tasks
1. [PQOS](https://github.com/intel/intel-cmt-cat) to generate real-time statistics of hardware counters


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
1. `--with-boost` to specify the prefix of the Boost installation
1. `--with-libunwind=prefix` to specify the prefix of the libunwind installation
1. `--with-papi=prefix` to specify the prefix of the PAPI installation
1. `--with-libnuma=prefix` to specify the prefix of the numactl installation
1. `--with-extrae=prefix` to specify the prefix of the extrae installation
1. `--enable-cuda` to enable support for CUDA tasks
1. `--with-pqos=prefix` to specify the prefix of the PQoS installation
1. `--enable-monitoring` to enable monitoring and predictions of task/CPU/thread statistics
1. `--enable-chrono-arch` to enable an architecture-based timer for the monitoring infrastructure
1. `--enable-monitoring-hwevents` to enable monitoring of hardware counters (which must be paired with an appropriate library)

The location of elfutils and hwloc is always retrieved through pkg-config.
The location of PAPI can also be retrieved through pkg-config if it is not specified through the `--with-papi` parameter.
If they are installed in non-standard locations, pkg-config can be told where to find them through the `PKG_CONFIG_PATH` environment variable.
For instance:

```sh
$ export PKG_CONFIG_PATH=$HOME/installations-mn4/elfutils-0.169/lib/pkgconfig:/apps/HWLOC/2.0.0/INTEL/lib/pkgconfig:$PKG_CONFIG_PATH
```

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


## Tracing, debugging and other options

Nanos6 applications, unlike Nanos++ applications do not require recompiling their code to generate extrae traces or to generate additional information.
This is instead controlled through environment variables, _envar_ from now on, at run time.


### Generating extrae traces

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


### Sample-based profiling

To enable sample-based profiling, run the application with the `NANOS6` envar set to `profile`.

In this mode, the runtime records backtraces of the threads up to a given depth and with a given frequency.
These parameters can be set through the following envars:

<table><tbody><tr><td> <strong>Name</strong> </td><td> <strong>Default value</strong> </td><td> <strong>Description</strong> 
</td></tr><tr><td> <em>NANOS6_PROFILE_NS_RESOLUTION</em> </td><td> 1000 </td><td> Sampling interval in nanoseconds 
</td></tr><tr><td> <em>NANOS6_PROFILE_BACKTRACE_DEPTH</em> </td><td> 4 </td><td> Number of stack frames to collect (excluding inlines) in each sample. 
</td></tr><tr><td> <em>NANOS6_PROFILE_BUFFER_SIZE</em> </td><td> 1000000000 </td><td> Number of sampling events to preallocate together in a chunk. The default value corresponds to 1 second of samples. 
</td></tr></tbody></table>

At the end of the execution, the runtime generates four files that contain entries sorted by decreasing frequency.
Their first column contains the sample count, and the rest, the actual entry values.
Their contents are the following:

__line-profile-PID.txt__: Source code lines

__function-profile-PID.txt__: Function names

__inline-profile-PID.txt__: Function names and source code lines including inlines
> Since the sampling is performed over the return addresses in the stack, if the compiler performs inlining, a given address can correspond to several functions. This file shows for the number of samples that have the same associated source code lines.


__backtrace-profile-by-line-PID.txt__: Function names and source code lines including inlines of a full backtrace
> Shows the number of samples that have a full backtrace that corresponds to the same exact source code lines.


__backtrace-profile-by-address-PID.txt__: Function names and source code lines including inlines of a full backtrace
> Shows the number of samples that have a full backtrace with the same exact return addresses.


When compiling, Mercurium performs transformations to the original source code.
At this time, Mercurium cannot preserve the original source code lines and function names.
Hence, the outputs of the profiler are based on the transformed code.
However, the transformed source code can be preserved by passing the `-keep` parameter to Mercurium.

Mercurium generates additional functions that wrap the task code.
These appear in the backtraces and their names begin with `nanos6_ol_` and `nanos6_unpack_` and are followed by a number.


### Obtaining statistics

To enable collecting statistics, run the application with the `NANOS6` envar set to either `stats` or `stats-papi`.
The first collects timing statistics and the second also records hardware counters.

By default, the statistics are emitted standard error when the program ends.
The output can be sent to a file through the `NANOS6_STATS_FILE` envar.

The contents of the output contains the average for each task type and the total task average of the following metrics:

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
Dependency Implementation linear-regions-fragmented
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
#	string	dependency_implementation	linear-regions-fragmented		Dependency Implementation
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
* `NANOS6_WISDOM_PATH`: To specify a path in which the wisdom file is located (to load and/or store metrics). By default `.nanos6_monitoring_wisdom.json`.



## Hardware Counters

As well as the Monitoring Infrastructure and the `stats-papi` version of the runtime, Nanos6 offers a real-time API for Hardware Counter statistics.
This API allows to obtain and predict hardware counters for tasks, similarly to Monitoring.

By default, the NULL option is used, which records no hardware counters.
At configure time, however, several options may be used to enable hardware counter monitoring using third party libraries.
Enabling this API is as easy as configuring the runtime with the `enable-monitoring-hwevents` option.

At this moment, Nanos6 offers intel-cmt-cat or PQoS to both obtain and generate metrics and predictions.
This library is enabled like so: `--with-pqos=prefix` where `prefix` specifies the installation path of PQoS.

The Hardware Counters API is controlled further with the following environment variables:
* `NANOS6_HARDWARE_COUNTERS_ENABLE`: To enable/disable hardware counter monitoring.
* `NANOS6_HARDWARE_COUNTERS_VERBOSE`: To enable/disable the verbose mode for hardware counter monitoring.
* `NANOS6_HARDWARE_COUNTERS_VERBOSE_FILE`: To specify an output file name to report hardware counter statistics.


## Cluster support

In order to enable OmpSs-2@Cluster support, you need a working MPI installation in your environment that supports multithreading, i.e. `MPI_THREAD_MULTIPLE`.
Nanos6 needs to be configured with the `--enable-cluster` flag.
For more information, on how to write and run cluster applications see [README-CLUSTER.md](docs/cluster/README-CLUSTER.md).
