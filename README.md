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

It's highly recommended to have an installation of the [OmpSs-2 LLVM/Clang](https://github.com/bsc-pm/llvm) which supports the OmpSs-2 model. When installing OmpSs-2 for the first time, you can break the chicken and egg dependency between Nanos6 and LLVM/Clang in the following way: you can build Nanos6 without specifying any LLVM/Clang, then, build LLVM/Clang specifying that Nanos6 installation, and finally, re-configure and build Nanos6 passing the `--with-nanos6-clang` to specify the LLVM/Clang path.

**Important:** The [Mercurium](https://github.com/bsc-pm/mcxx) source-to-source compiler is the OmpSs-2 legacy compiler and it is unsupported now. You do not have to install it to use OmpSs-2. We recommend using the [LLVM/Clang](https://github.com/bsc-pm/llvm) compiler instead.

### Optional libraries and tools

In addition to the build requirements, the following libraries and tools enable additional features:

1. [Extrae](https://tools.bsc.es/extrae) to generate execution traces for offline performance analysis with [Paraver](https://tools.bsc.es/paraver)
1. [elfutils](https://sourceware.org/elfutils/) and [libunwind](http://www.nongnu.org/libunwind) to generate sample-based profiling
1. [graphviz](http://www.graphviz.org/) and pdfjam or pdfjoin from [TeX](http://www.tug.org/texlive/) to generate graphical representations of the dependency graph
1. [parallel](https://www.gnu.org/software/parallel/) to generate the graph representation in parallel
1. [CUDA](https://developer.nvidia.com/cuda-zone) to enable CUDA tasks
1. [PGI or NVIDIA HPC-SDK](https://pgroup.com) to enable OpenACC tasks
1. [PQOS](https://github.com/intel/intel-cmt-cat) to generate real-time statistics of hardware counters
1. [DLB](https://pm.bsc.es/dlb) to enable dynamic management and sharing of computing resources
1. [jemalloc](https://github.com/jemalloc/jemalloc) to use jemalloc as the default memory allocator, providing better performance than the default glibc implementation. Jemalloc must be compiled with `--enable-stats` and `--with-jemalloc-prefix=nanos6_je_` to link with the runtime
1. [PAPI](http://icl.utk.edu/papi/software/) >= 5.6.0
1. [Babeltrace2](https://babeltrace.org/) to enable the fast CTF converter (`ctf2prv --fast`) and the multi-process trace merger (`nanos6-mergeprv`)
1. [ovni](https://ovni.readthedocs.io/) to generate execution traces for performance analysis with [Paraver](https://tools.bsc.es/paraver)


## Build procedure

Nanos6 uses the standard GNU automake and libtool toolchain.
When cloning from a repository, the building environment must be prepared through the following command:

```sh
$ ./autogen.sh
```

Use this script instead of the `autoreconf` command. When the code is distributed through a tarball, it usually does not need that command.

Then execute the following commands:

```sh
$ ./configure --prefix=INSTALLATION_PREFIX ...other options...
$ make all check
$ make install
```

where `INSTALLATION_PREFIX` is the directory into which to install Nanos6.

The configure script accepts the following options:

1. `--with-nanos6-clang=prefix` to specify the prefix of the LLVM/Clang installation which supports OmpSs-2
1. `--with-nanos6-mercurium=prefix` to specify the prefix of the Mercurium installation
1. `--with-boost=prefix` to specify the prefix of the Boost installation
1. `--with-libunwind=prefix` to specify the prefix of the libunwind installation
1. `--with-libnuma=prefix` to specify the prefix of the numactl installation
1. `--with-extrae=prefix` to specify the prefix of the extrae installation
1. `--with-dlb=prefix` to specify the prefix of the DLB installation
1. `--with-jemalloc=prefix` to specify the prefix of the jemalloc installation
1. `--with-papi=prefix` to specify the prefix of the PAPI installation
1. `--with-pqos=prefix` to specify the prefix of the PQoS installation
1. `--with-cuda[=prefix]` to enable support for CUDA tasks; optionally specify the prefix of the CUDA installation, if needed
1. `--enable-openacc` to enable support for OpenACC tasks; requires PGI compilers
1. `--with-pgi=prefix` to specify the prefix of the PGI or NVIDIA HPC-SDK compilers installation, in case they are not in `$PATH`
1. `--enable-chrono-arch` to enable an architecture-based timer for the monitoring infrastructure
1. `--with-babeltrace2=prefix` to specify the prefix of the Babeltrace2 installation and enable the fast CTF converter (`ctf2prv --fast`) and the multi-process trace merger (`nanos6-mergeprv`)
1. `--with-ovni=prefix` to specify the prefix of the ovni installation and enable the ovni instrumentation

The hwloc dependency can be specified using the `--with-hwloc` option. This option can take these values:
* `pkgconfig`: The hwloc is an external installation and Nanos6 should discover it through the pkg-config tool.
Make sure to set the `PKG_CONFIG_PATH` if the hwloc is not installed in non-standard directories.
This is the **default** behavior if the option is not present or no value is provided
* A prefix of an external hwloc installation
* `embedded`: The hwloc is built and embedded into the Nanos6 library as an internal module.
This is useful when user programs may have third-party software (e.g., MPI libraries) that depend on a different hwloc version and may conflict with the one used by Nanos6.
When embedded, the hwloc library is internal and is only used by Nanos6.
To this end, the `deps` folder contains a default hwloc source tarball.
This tarball is automatically extracted into `deps/hwloc` by our `autogen.sh` script, which is then built and embedded when `--with-hwloc=embedded` is chosen.
You may change the embedded hwloc version by placing the desired tarball inside the `deps` folder and re-running `autogen.sh` with the option `--embed-hwloc <VERSION>`.
For the moment, the tarball must follow the format `deps/hwloc-<VERSION>.tar.gz`

The location of elfutils is always retrieved through pkg-config.
The same occurs for hwloc by default or when specifying `--with-hwloc=pkgconfig`.
If they are installed in non-standard locations, pkg-config can be told where to find them through the `PKG_CONFIG_PATH` environment variable.
For instance:

```sh
$ export PKG_CONFIG_PATH=$HOME/installations-mn4/elfutils-0.169/lib/pkgconfig:/apps/HWLOC/2.0.0/INTEL/lib/pkgconfig:$PKG_CONFIG_PATH
```

To enable CUDA the `--with-cuda` flag is needed.
The location of CUDA can be retrieved automatically, if it is in standard system locations (`/usr/lib`, `/usr/include`, etc), or through pkg-config.
Alternatively, for non-standard installation paths, it can be specified using the optional `=prefix` of the parameter.

The ``--enable-openacc`` flag is needed to enable OpenACC tasks.
The location of PGI compilers can be retrieved from the `$PATH` variable, if it is not specified through the `--with-pgi` parameter.

After Nanos6 has been installed, it can be used by compiling your C and C++ codes with LLVM/Clang using the `-fompss-2` flag.
Example:

```sh
$ clang -c -fompss-2 a_part_in_c.c
$ clang++ -c -fompss-2 a_part_in_c_plus_plus.cxx
$ clang++ -fompss-2 a_part_in_c.o a_part_in_c_plus_plus.o -o app
```

We still recommend using the [Mercurium](https://github.com/bsc-pm/mcxx) source-to-source compiler for Fortran applications.

## Execution

Nanos6 applications can be executed as is.
The number of cores that are used is controlled by running the application through the `taskset` command.
For instance:

```sh
$ taskset -c 0-2,4 ./app
```

would run `app` on cores 0, 1, 2 and 4.

### Runtime settings

The behaviour of the Nanos6 runtime can be tuned after compilation by means of a configuration file.
The default configuration file `nanos6.toml` can be found in the documentation directory of the Nanos6 installation, usually in the `$INSTALLATION_PREFIX/share/doc/nanos6/scripts` folder.
Currently, the supported format is TOML v1.0.0-rc1 (https://toml.io/en/v1.0.0-rc.1).

To override the default configuration, we recommended to copy the default file and change the relevant options.
The Nanos6 runtime will only interpret the first configuration file found according to the following order:

1. The file pointed by the `NANOS6_CONFIG` environment variable.
1. The file `nanos6.toml` found in the current working directory.
1. The file `nanos6.toml` found in the installation path (default file).

Alternatively, if configuration has to be changed programatically and creating new files is not practical, configuration variables can be overriden using the `NANOS6_CONFIG_OVERRIDE` environment variable.
The contents of this variable have to be in the format `key1=value1,key2=value2,key3=value3,...`.
For example, to change the dependency implementation and CTF instrumentation: `NANOS6_CONFIG_OVERRIDE="version.dependencies=discrete,version.instrument=ctf" ./ompss-program`.

### Scheduling options

The scheduling infrastructure provides the following configuration variables to modify the behavior of the task scheduler.

* `scheduler.policy`: Specifies whether ready tasks are added to the ready queue using a FIFO (`fifo`) or a LIFO (`lifo`) policy. The **fifo** is the default.
* `scheduler.immediate_successor`: Probability of enabling the immediate successor feature to improve cache data reutilization between successor tasks. If enabled, when a CPU finishes a task it starts executing the successor task (computed through their data dependencies). Default is **0.75**.
* `scheduler.priority`: Boolean indicating whether the scheduler should consider the task priorities defined by the user in the task's priority clause. **Enabled** by default.

## Benchmarking, tracing, debugging and other options

There are several Nanos6 variants, each one focusing on different aspects of parallel executions: performance, debugging, instrumentation, etc.
Nanos6 applications, unlike Nanos++ applications do not require recompiling their code to generate Extrae traces or to generate additional information.
This is instead controlled through configration options, at run time.
Users can select a Nanos6 variant when running an application through the `version.dependencies`, `version.instrument` and `version.debug` configuration variables.
The next subsections explain the different variants of Nanos6 and how to enable them.

### Benchmarking

The default variant is the optimized one, which disables the `version.debug` (no debug) and the `version.instrument` to `none` (no instrumentation).
This is compiled with high optimization flags, it does not perform validity checks and does not provide debug information.
That is the variant that should be used when performing benchmarking of parallel applications.

Additionally, Nanos6 offers an extra performant option named `turbo`, which is the same as `optimized` but adding further optimizations.
It enables two Intel® floating-point (FP) unit optimizations in all tasks: flush-to-zero (FZ) and denormals are zero (DAZ).
Please note these FP optimizations could alter the precision of floating-point computations.
It is disabled by default, but can be enabled by setting the `turbo.enabled` configuration option to `true`.

Moreover, these variants can be combined with the jemalloc memory allocator (``--with-jemalloc``) to obtain the best performance.
Changing the dependency system implementation may also affect the performance of the applications.
The different dependency implementations and how to enable them are explained in the Section [Choosing a dependency implementation](#choosing-a-dependency-implementation).


### Tracing an OmpSs-2 application with ovni (recommended)

Nanos6 can generate execution traces with the ovni library, which generates
lightweight binary traces, and it is possible to mix ovni-instrumented libraries
together with an OmpSs-2 program and obtain a single coherent trace.

To enable the generation of ovni traces, Nanos6 must be built with the
`--with-ovni` option, and without `--disable-ovni-instrumentation`. The
application must run with the `version.instrument=ovni` configuration option.
The trace will be left in a `ovni/` directory, which can be transformed into a
Paraver trace with the `ovniemu` utility. The Paraver configuration files
(views) can be found in the `ovni/cfg` directory.

See the [ovni documentation][ovnidoc] for more details.

[ovnidoc]: https://ovni.readthedocs.io/

The level of detail can be controlled with the `instrument.ovni.level`
configuration option, a higher number includes more events but also incurs in a
larger performance penalty.


### Tracing an OmpSs-2 application with Extrae

To generate an extrae trace, run the application with the `version.instrument` config set to `extrae`.

Currently there is an incompatibility when generating traces with PAPI.
To solve it, define the following config: `instrument.extrae.as_threads = true`

The resulting trace will show the activity of the actual threads instead of the activity at each CPU.
In the future, this problem will be fixed.


### Tracing an OmpSs-2 application with CTF

Nanos6 includes another instrumentation mechanism which provides detailed
information of the internal runtime state as the execution evolves. The
instrumentation produces a lightweight binary trace in the CTF format which is
later converted to the Paraver PRV format. To generate a trace, run the
application with `version.instrument=ctf`.

By default, only Nanos6 internal events are recorded, such as the information
about the tasks or the state of the runtime. For details on how to additionally
record system-wide Linux Kernel events, please check the section "Linux Kernel
Tracing" under [CTF.md](docs/ctf/CTF.md).

The main trace directory named `trace_<binary_name>` will be created at the current
working directory at the end of the execution, which contains all the related
trace files and directories.

The CTF instrumentation supports multiple processes running in parallel with MPI.
In order to coordinate the clock synchronization, it is required to run the
application with [TAMPI](https://github.com/bsc-pm/tampi) (at least version 1.1).

Every process will create the rank subdirectory inside the trace directory, with
a name that corresponds to the rank number. In the absence of MPI, when there is
only a single process, the folder will be named 0.

Inside the rank directory, the CTF trace is stored in a subdirectory named
"ctf". A post-processing step is required to reconstruct the timeline of events
from the CTF trace. In order to visualize the events, the trace is converted to
the Paraver PRV format. The resulting PRV trace is stored in the "prv"
subdirectory.

By default, Nanos6 will convert the trace automatically at the end of the
execution unless the user explicitly sets the configuration variable
`instrument.ctf.converter.enabled = false`.

The environment variable `CTF2PRV_TIMEOUT=<minutes>` can be set to stop the
conversion after the specified elapsed time in minutes. Please note that the
conversion tool requires python3 and the babeltrace2 packages.

An experimental conversion tool written in C is included, with a faster
conversion speed, but not all features are yet supported. In order to
enable it, Nanos6 must be compiled with babeltrace2 support using the configure
option `--with-babeltrace2=prefix`, pointing to a valid babeltrace2 installation.
Additionally, you will need to enable the fast converter in the configuration
with `instrument.ctf.converter.fast = true`.

Every Nanos6 process will only convert its own CTF trace to PRV. When you have
multiple MPI processes, you may want to integrate all the PRV files per rank
into a single trace. Beware that it may easily exceed the recommended PRV size
for Paraver. You can use the included merger as:

	$ nanos6-mergeprv trace_<binary_name>

The merged trace will be placed in the main trace directory, at
`trace_<binary_name>/trace.prv`.
Please take into account that the `nanos6-mergeprv` can only merge traces generated
by the fast CTF converter.

#### Paraver configurations for CTF

The Paraver configuration files can be found under:

```sh
$INSTALLATION_PREFIX/share/doc/nanos6/paraver-cfg/nanos6/ctf2prv/
```

Please, note that ctf2prv views are not compatible with Extrae traces and vice-versa.

Additionally, the following command can be used to convert a trace manually:

```sh
$ ctf2prv $TRACE
```

which will generate the directory `$TRACE/prv` with the Paraver trace.

Although the `ctf2prv` tool requires python3 and babeltrace2 python modules,
Nanos6 does not require any package to generate CTF traces.  For more
information on how the CTF instrumentation variant works see
[CTF.md](docs/ctf/CTF.md).

To run the experimental fast converter, add the option `--fast`.


### Generating a graphical representation of the dependency graph

To generate the graph, run the application with the `version.instrument` config set to `graph`.

By default, the graph nodes include the full path of the source code.
To remove the directories, set the `instrument.graph.shorten_filenames` config to `true`.

The resulting file is a PDF that contains several pages.
Each page represents the graph at a given point in time.
Setting the `instrument.graph.show_dead_dependencies` config to `true` forces future and previous dependencies to be shown with different graphical attributes.

The `instrument.graph.display` config, if set to `true`, will make the resulting PDF to be opened automatically.
The default viewer is `xdg-open`, but it can be overridden through the `instrument.graph.display_command` config.

For best results, we suggest to display the PDF with "single page" view, showing a full page and to advance page by page.


### Verbose logging

To enable verbose logging, run the application with the `version.instrument` config set to `verbose`.

By default it generates a lot of information.
This is controlled by the `instrument.verbose.areas` config, which can contain a list of areas.
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
For instance, `areas = [ "AddTask", "TaskExecution", "TaskWait" ]` is a good starting point.

By default, the output is emitted to standard error, but it can be sent to a file by specifying it through the `instrument.verbose.file` config.
Also `instrument.verbose.dump_only_on_exit` can be set to `true` to delay the output to the end of the program to avoid getting it mixed with the output of the program.


### Obtaining statistics

To enable collecting timing statistics, run the application with the `version.instrument` config set to `stats`.

By default, the statistics are emitted standard error when the program ends.
The output can be sent to a file through the `instrument.stats.output_file` config.

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
To enable validity checks, run the application with the `version.debug` config set to `true`.
This will enable many internal validity checks that may be violated with the application code is incorrect.
In the future we may include a validation mode that will perform extensive application code validation.
Notice that all instrumentation variants can be executed either with or without enabling the debug option.

To debug dependencies, it is advised to reduce the problem size so that very few tasks trigger the problem, and then use let the runtime make a graphical representation of the dependency graph as shown previously.

Processing the configuration file involves selecting at run time a runtime compiled for the corresponding instrumentation.
This part of the bootstrap is performed by a component of the runtime called "loader".
To debug problems due to the installation, run the application with the `loader.verbose` config variable set to `true`.

## Runtime information

Information about the runtime may be obtained by running the application with the `loader.report_prefix` config set, or by invoking the following command:

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

The `loader.report_prefix` config may contain a string that will be prepended to each line.
For instance, it can contain a sequence that starts a comment in the output of the program.
Example:

```sh
$ NANOS6_CONFIG_OVERRIDE="loader.report_prefix=#" ./app
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

Monitoring is an infrastructure composed of several modules that gather metrics of specific elements of the runtime core and generate predictions usable by other modules. Checkpointing of predictions is enabled through the Wisdom mechanism, which allows saving predictions for future executions.

Monitoring is enabled at run-time through various configuration variables:

* `monitoring.enabled`: To enable/disable monitoring, disabled by default.
* `monitoring.verbose`: To enable/disable the verbose mode for monitoring. Enabled by default if monitoring is enabled.
* `monitoring.rolling_window`: To specify the number of metrics used for accumulators (moving average's window). By default, the latest 20 metrics.

Additionally, checkpointing of predictions is enabled through the `Wisdom` mechanism, which allows saving normalized metrics for future executions. It is controlled by the following configuration variable:

* `monitoring.wisdom`: To enable/disable the wisdom mechanism. Disabled by default.


## Hardware Counters

Nanos6 offers an infrastructure to obtain hardware counter statistics of tasks with various backends. The usage of this API is controlled through the Nanos6 configure file. Currently, Nanos6 supports the PAPI, RAPL and PQoS backends.

All the available hardware counter backends are listed in the default configuration file, found in the scripts folder. To enable any of these, modify the `false` fields and change them to `true`. Specific counters can be enabled or disabled by adding or removing their name from the list of counters inside each backend subsection.

Next we showcase a simplified version of the hardware counter section of the configure file, where the PAPI backend is enabled with counters that monitor the total number of instructions and cycles, and the PAPI backend is enabled as well:

```toml
[hardware_counters]
  [hardware_counters.papi]
    enabled = true
    counters = ["PAPI_TOT_INS", "PAPI_TOT_CYC"]
  [hardware_counters.rapl]
    enabled = true
```

## Device tasks

For information about using device tasks (e.g., CUDA tasks), refer to the [devices](docs/devices/Devices.md) documentation.

## Choosing a dependency implementation

The Nanos6 runtime has support for different dependency implementations. The `discrete` dependencies are the default dependency implementation. This is the most optimized implementation but it does not fully support the OmpSs-2 dependency model since it does not support region dependencies. In the case the user program requires region dependencies (e.g., to detect dependencies among partial overlapping dependency regions), Nanos6 privides the `regions` implementation, which is completely spec-compliant.

The dependency implementation can be selected at run-time through the `version.dependencies` configuration variable. The available implementations are:

* `version.dependencies = "discrete"`: Optimized implementation not supporting region dependencies. Region syntax is supported but will behave as a discrete dependency to the first address. Scales better than the default implementation thanks to its simpler logic and is functionally similar to traditional OpenMP model. **Default** implementation.
* `version.dependencies = "regions"`: Supporting all dependency features.

In case an OmpSs-2 program requires region dependency support, it is recommended to add the declarative directive below in any of the program source files. Then, before the program is started, the runtime will check whether the loaded dependency implementation is `regions` and will abort the execution if it is not true.

```c
#pragma oss assert("version.dependencies==regions")

int main() {
	// ...
}
```

Notice that the assert directive could also check whether the runtime is using `discrete` dependencies. The directive supports conditions with the compare operators `==` and `!=`.


## DLB Support

DLB is a library devoted to speed up hybrid parallel applications and maximize the utilization of computational resources. More information about this library can be found [here](https://pm.bsc.es/dlb). To enable DLB support for Nanos6, a working DLB installation must be present in your environment. Configuring Nanos6 with DLB support is done through the `--with-dlb` flag, specifying the root directory of the DLB installation.

After configuring DLB support for Nanos6, its enabling can be controlled at run-time through the `dlb.enabled` configuration variable. To run with Nanos6 with DLB support then, this variable must be set to true, since by default DLB is disabled.

Once DLB is enabled for Nanos6, OmpSs-2 applications will benefit from dynamic resource sharing automatically. The following example showcases the executions of two applications that share the available CPUs between them:

```sh
# Run the first application using 10 CPUs (0, 1, ..., 9)
taskset -c 0-9   ./merge-sort.test &

# Run the second application using 10 CPUs (10, 11, ..., 19)
taskset -c 10-19 ./cholesky-fact.test &

# At this point the previous applications should be running while sharing resources
# ...
```

## Polling Capabilities

The polling services API is no longer supported and has been replaced by another mechanism more efficient and flexible.
Now the polling feature is provided by a regular task scheduled periodically thanks to the `nanos6_wait_for` API function.
The function, shown below, blocks the calling task during `time_us` microseconds (approximately), and the runtime system uses the CPU to execute other ready tasks meanwhile.

```c
uint64_t nanos6_wait_for(uint64_t time_us);
```

The function returns the actual time that has been sleeping, so the caller can take decisions based on that.
Notice that the polling frequency is now dynamic and can be set programmatically.
To implement a polling task, we recommend spawning a function using the `nanos6_spawn_function`, which instantiates an isolated task with an independent namespace of data dependencies and no relationship with others task (i.e. no taskwait will wait for it).

## CPU Managing Policies

Currently, Nanos6 offers different policies when handling CPUs through the `cpumanager.policy` configuration variable:
* `cpumanager.policy = "idle"`: Activates the `idle` policy, in which idle threads halt on a blocking condition, while not consuming CPU cycles.
* `cpumanager.policy = "busy"`: Activates the `busy` policy, in which idle threads continue spinning and never halt, consuming CPU cycles.
* `cpumanager.policy = "hybrid"`: Activates the `hybrid` policy, in which idle threads spin for a specific number of iterations before halting on a blocking condition. The number of iterations is controlled by the `cpumanager.busy_iters` configuration variable, which defaults to 240000 collective iterations across all the available CPUs (the real number per CPU is the collective one divided by the number of CPUs).
* `cpumanager.policy = "lewi"`: If DLB is enabled, activates the LeWI policy. Similarly to the idle policy, in this one idle threads lend their CPU to other runtimes or processes.
* `cpumanager.policy = "greedy"`: If DLB is enabled, activates the `greedy` policy, in which CPUs from the process' mask are never lent, but allows acquiring and lending external CPUs.
* `cpumanager.policy = "default"`: Fallback to the default implementation. If DLB is disabled, this policy falls back to the `hybrid` policy, while if DLB is enabled it falls back to the `lewi` policy.

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
Although the throttle feature is disabled by default, it can be enabled and tunned at runtime through the following configuration variables:

* `throttle.enabled`: Boolean variable that enables the throttle mechanism. **Disabled** by default.
* `throttle.tasks`: Maximum absolute number of alive childs that any task can have. It is divided by 10 at each nesting level. By default is 5.000.000.
* `throttle.pressure`: Percentage of memory budget used at which point the number of tasks allowed to exist will be decreased linearly until reaching 1 at 100% memory pressure. By default is 70.
* `throttle.max_memory`: Maximum used memory or memory budget. Note that this variable can be set in terms of bytes or in memory units. For example: ``throttle.max_memory = "50GB"``. The default is the half of the available physical memory.

## NUMA support

Nanos6 includes NUMA support based on three main components: an allocation/deallocation API, a data tracking system and a locality-aware scheduler.
When allocating memory using the nanos6 NUMA API, we annotate in our directory the location of the data. Then, when a task becomes ready, we check where is each of the data dependences of the task, and schedule the task to be run in the NUMA node with a greater share of its data. The NUMA support can be handled through the `numa.tracking` configuration variable:
* `numa.tracking = "on"`: Enables the NUMA support.
* `numa.tracking = "off"`: Disables the NUMA support.
* `numa.tracking = "auto"`: The NUMA support is enabled in the first allocation done using the Nanos6 NUMA API. If no allocation is done, the support is never enabled.

## Cluster support

This reference implementation of the Nanos6 runtime system does no longer support the OmpSs-2@Cluster programming model.
Please check the [Nanos6 Cluster repository](https://github.com/bsc-pm/nanos6-cluster) for a stable variant supporting OmpSs-2@Cluster.
