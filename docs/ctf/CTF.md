Introduction
============

The CTF instrumentation backend is a performance evaluation and debugging tool aimed for Nanos6 users and developers alike.
Its purpose is to record the most relevant information of a Nanos6 execution efficiently and to provide means for visually inspecting it offline.
The Nanos6 CTF backend is simple (no installation prerequisites) and mostly lock-free (see the section "Implementation details" for more information).

Nanos6 stores traces in the Common Trace Format (CTF).
CTF traces can be directly visualized with tools such as babeltrace1 or babeltrace2 (raw command line inspector).
But it is recommended first to convert them to Paraver traces using the provided `ctf2prv` command.
Although Nanos6 requires no special packages to write CTF traces, the `ctf2prv` converter needs python3 and the babeltrace2 python bindings.

Nanos6 can simultaneously collect Linux Kernel events and store them in CTF format.
Linux kernel events provide system-wide information such as context switches and interrupts, among others.
Enabling Linux Kernel events tracing requires special permissions (see the section "Kernel tracing" for more details).
The advantage of using the Nanos6 Linux Kernel tracing facility compared to other tracing solutions is its integration:
Linux Kernel events are collected when the Nanos6 core deems appropriate, minimizing the disturbance on the application execution flow.
Also, Nanos6 provides a simple interface based on presets to select the most relevant Linux Kernel events but also allows more experienced users to provide a list of manually defined events.

The Nanos6 CTF backend will transparently collect hardware counters information if the Nanos6 Hardware counters infrastructure has been enabled.
See the Nanos6 README for more information on how to enable Hardware Counters.

This backend objective is not to replace Extrae, but to provide a minimum set of features for easy Nanos6 introspection.
More sophisticated analysis (such as task dependency graphs) will have to be performed with Extrae as usual.
See [README.md](README.md) for Nanos6 and Extrae usage.

Implementation details
----------------------

Nanos6 keeps a per-core circular buffer to store CTF events before flushing them to the storage device.
Each Nanos6 worker threads is pinned to a core and uses the core's buffer to store the worker's emitted events.
Therefore, emitting an event from a worker thread is a lock-free operation.
However, events emitted by external threads (threads not controlled by Nanos6 and possibly not pinned to any core) are stored in a shared buffer and it is necessary to take a lock for writing an event.

When Linux Kernel events are enabled, a second per-core buffer is allocated to store kernel-provided events.
Internally, kernel events are collected using the `perf_event_open()` system call, which requires to open a file descriptor per-core and per-event.
The `perf_event_open` syscall exports events through a shared per-core memory mapped circular buffer.
Nanos6 periodically moves events from the kernel circular buffer to the per-core nanos6 buffer, formatting the event has necessary during the process.
However, is worth noting that kernel events might be lost if too many events are generated before Nanos6 can move them.

Nanos6 event circular buffers are divided into four subbuffers.
Filled subbuffers are flushed to the storage device by worker threads in between tasks executions.
Flushing is done synchronously (i.e. while a worker is performing a flush, it will not perform other tasks) and it might affect the execution workflow.
External threads (but also worker threads) will flush their buffers when the next generated event does not fit into the buffer.
Flushing operations are also recorded and can be inspected with the "CTF flush buffers to disk" view.

Traces are written by default under the `instrument.ctf.tmpdir` configuration variable or under `/tmp` if not set.
Traces written to `/tmp` are kept in RAM memory (see tmpfs for more information) and flushing translates to a memory copy operation.
When an application execution finishes, Nanos6 copies the trace to the current directory.

Usage
=====

To generate a CTF trace, run the application with the `version.instrument` config set to `ctf`.

This will create a `trace-<app_name>-<app_pid>` folder in the current directory, hereinafter refered to as `$TRACE` for convenience.
The subdirectory `$TRACE/ctf` contains the ctf trace as recorded by Nanos6.

By default, Nanos6 will convert the trace automatically at the end of the execution unless the user explicitly sets the configuration variable `instrument.ctf.conversor.enabled = false`.
The converted Paraver trace will be stored under the `$TRACE/prv` subdirectory.
The environment variable `CTF2PRV_TIMEOUT=<minutes>` can be set to stop the conversion after the specified elapsed time in minutes.
Please note that the conversion tool requires python3 and the babeltrace2 package.

Additionally, Nanos6 provides a command to manually convert traces:

```sh
$ ctf2prv $TRACE
```

which will generate the directory `$TRACE/prv` with the Paraver trace.


Linux Kernel Tracing
==============

Requirements:
 - Linux Kernel >= 4.1.0
 - Set /proc/sys/kernel/perf\_event\_paranoid to -1
 - Grant the current user read permissions for /sys/kernel/debug/tracing or /sys/kernel/tracing (tracefs standard mountpoints)

The Linux Kernel provides a set of events (also named tracepoints).
To enable kernel events tracing in Nanos6, the user needs to specify which events wants to collect through the nanos6 configuration file.
The option `instrument.ctf.events.kernel.presets` allows to speciy a list of presets that ease selecting events for supported Paraver views.
For instance, try to set `instrument.ctf.events.kernel.presets=[preemption]` to enable preemption-related events.

Please check the Paraver views description below for the list of presets that each view requries.

If the user does not want to enable some of the events defined in a preset, it can blacklist them in the `instrument.ctf.events.kernel.exclude` list.

Additionally, the user can provide a file path in `instrument.ctf.events.kernel.file` with a list of raw kernel events to enable.
An example file named `kernel_events.conf` follows.

```
$ cat kernel_events.conf
sched_switch
sys_enter_open
sys_exit_open
# mm_page_alloc # comented lines are not enabled
```

A list of events that your system supports can be obtained with:

```sh
perf list tracepoint | grep Tracepoint | awk '{sub(/.*:/,"",$1); print $1}'
```

Or by inspecting the contents of tracefs, usually mounted by default at `/sys/kernel/tracing/events` or `/sys/kernel/debug/tracing/events`.
Also, Nanos6 creates the file `$TRACE/nanos6_kerneldefs.json` which contains a human-readable format definition of all supported system events.
Unfortunately, there is no official Linux description of each event at the moment; please check the Linux Kernel source code for a definition of the events you are interested into.

If too many events are enabled, tracing might fail due to reaching the limit of open files in the system (one file descriptor is opened per event and core).
In that case, please, increase the open file limit, reduce the number of events and/or the number of cores.

The CTF kernel trace is stored under `$TRACE/ctf/kernel` whilst the Nanos6 internal events trace are stored in `$TRACE/ctf/ust`.
Nanos6 will convert the generated CTF kernel trace (both user and kernel) into Paraver at the end of the execution, as usual.

Paraver Views
=============

A number of Paraver views are provided with each Nanos6 distribution under the directory:

```sh
$NANOS6_INSTALL/share/doc/nanos6/paraver-cfg/ctf2prv
```

Some views refer to "Nanos6 core code".
This encompasses all code that it is not strictly user-written code (task code).
This includes, for instance, the worker thread main loop (where tasks are requested constantly) the Nanos6 initialization code or the shutdown code.

The Paraver minimum resolution is 1 pixel.
Hence, if the current zoom level is not close enough, multiple events might fall into the same pixel.
In this situation, Paraver must choose which of these events must be colored.
Several visualization options to tackle this problem can be found under the "Drawmode" option in the context menu (right mouse click on a Paraver view).
By default, most views have the "Random" mode, which means that drawn events are selected randomly.
This is useful to avoid giving priority to an event or another, but the Paraver user must be aware that when zooming in (or out) events might seem to "move" due to another set of events being selected randomly after zooming.
In the particular case of graphs (such as the "Number of" views), the Drawmode "Maximum" option must be selected if searching for maximum peaks, otherwise the event with the maximum value might not be selected to be drawn and the peak might remain hidden.
Similarly, "Minimum" must be selected if searching for minimum peaks.

Please, note that some views might complain about missing events when opened in Paraver.
It is likely that the missing event is "Runtime: Busy Waiting" as it is only generated when running Nanos6 under the `busy` policy.
You can safely ignore the message.
If unsure, you can try disabling the "Runtime: Busy Waiting" in your view, save the cfg, and reload the cfg again.

Do not use the Extrae cfg's views as event identifiers are not compatible.

Tasks
-----

Shows the name (label) of tasks executed in each core.

Tasks and Runtime
-----------------

Shows the name (label) of tasks executed in each core.
It also displays the time spent on Nanos6 Runtime (shown as "Runtime") and threads busy waiting (shown as "Busy waiting") while waiting for ready tasks to execute.

Task Source Code
----------------

Shows the source code location of tasks executed in each core.
It also displays the time spent on Nanos6 Runtime (shown as "Runtime") and threads busy waiting (shown as "Busy waiting") while waiting for ready tasks to execute.

Hardware Counters
-----------------

Shows the collected hardware counters (HWC) information.
Please note that you must enable HWC collection in Nanos6 first.
See Nanos6 documentation for more details.

HWC information is collected in a per-task burst basis (a chunk of task execution without an interruption such as the task being blocked) and in a per-core (Nanos6) basis.
By default, it displays `PAPI_TOT_INS`.
Each HWC is displayed as a different Extrae Event.
If you want to inspect another counter, please, modify the "Value to display" subview of the "Hardware Counters" view to display the appropriate Extrae Event (HWC event) under Paraver's "Filter -> Events -> Event Type -> Types" menu.

Task Id
-------

Shows the unique id of the tasks executed in each core.

The default colouring of this view is "Not Null Gradient Color", which eases identifying the execution progress of tasks as they were created.
The id's 0, 1, 2 and 3 are reserved for Idle, Runtime, Busy Wait and Task, respectively.
Hence, it might be interesting to manually set the Paraver's "Semantic Minimum" value to 4, which will not include these values for the gradient colouring.
This is the value by default, but if you perform a semantic adjustment, you will need to change the "Semantic Minimum" manually again.

It might also be interesting to draw this view as "Code Color" which will make it easier to spot different consecutive tasks.
Under this colouring scheme, it is easier to see the time spent running Nanos6 core code shown as "Runtime" and when threads perform a "Busy waiting" while waiting for more tasks to execute.

Thread Id
---------

Shows the thread Id (TID) of the thread that was running in each core.
Be aware that this view only shows the thread placement according to Nanos6 perspective, not the OS perspective.
This means that even if this view shows a thread running uninterruptedly in a core for a long time, the system might have preempted the Nanos6 thread by another system thread a number of times without this being displayed in the view.
Please, use the "Linux Kernel Thread Ids" view to see system-wide TIDs.

Runtime Status Simple
---------------------

Shows the a simplified view of the runtime states.
It displays the time spent running Nanos6 core code shown as "Runtime", when threads perform a "Busy Waiting" while waiting for more tasks to execute and when the runtime is running task's code shown as "Task".

Runtime Subsystems
------------------

Shows the activity of the most interesting Nanos6 subsystems.
In essence, a coarse-grained classification of the time spent while running Nanos6 core code.
The displayed subsystems include: Task creation/initialization, dependency registration/unregistration, scheduling add/get tasks and polling services.

Number of Blocked Tasks
-----------------------

Shows a graph of tasks in the blocked state.
Blocked tasks are tasks that started running at some point and then stopped running before completing.
This might block, for instance, due to a taskwait directive.

You might want to change paraver's "Drawmode" option to "Maximum" or "Minimum" if searching for maximum or minimum peaks.
By default, it is set to "Maximum", which means that minimum peaks might be hidden if the view is too much zoomed out.

Number of Running Tasks
-----------------------

Shows a graph of tasks being executed by some worker thread.

Number of Created Tasks
-----------------------

Shows a graph with the count of total created tasks.
A task is created when its Nanos6 data structures are allocated and registered within the Nanos6 core.

Number of Created Workers
-------------------------

Shows a graph of created worker threads by Nanos6.
Once a worker thread is created, it is not destroyed until the end of the application's execution.

Number of Running Workers
-------------------------

Shows a graph of Nanos6 worker threads that are allowed to run on a core.
Please note that even if Nanos6 allows a worker to run, another system thread might have temporarily preempted that worker.
System preemptions are not displayed in this view.

Number of Blocked Workers
-------------------------

Shows a graph of worker threads that are blocked (no longer running), either because they are idle or because they are waiting for some event.
This view only counts workers that blocked due to Nanos6 will.
If a worker that is running a task blocks because of a blocking operation performed by the task code (hence, outside the scope of Nanos6) it will not be shown in this view.

CTF Flush Buffers to Disk
-------------------------

Eventually, the Nanos6 buffers that hold the events are filled and need to be flushed to disk.
When this happens, the application's execution might be altered because of the introduced overhead.
This flushing can occur in a number of places within the Nanos6 core, either because the buffers were completely full or because Nanos6 decided to flush them before reaching that limit.

This view shows exactly when the flushing happened.
If attempting to write the event A into the Nanos6 event buffer triggers a flush, the produced Nanos6 trace will first show the event that triggered the flush followed by the flushing events.

Paraver Views for Kernel Events
===============================

All the views listed below require Linux Kernel events.
Please, check the section "Linux Kernel Tracing" for more details.

Paraver Views that depend on kernel events can be found under:

```sh
$INSTALLATION_PREFIX/share/doc/nanos6/paraver-cfg/nanos6/ctf2prv/kernel
```

Kernel Preemptions
------------------

Uses Linux Kernel events to show preemptions affecting the cores where the traced application runs.
Preemptions include interruptions and both user and kernel threads.
Interrupts will be shown as "IRQ" and "Soft IRQ".
IRQs are run in interrupt context.
Soft IRQs are deferred interrupt work that run out of interrupt context.

This view requires the Linux Kernel "preemption" preset (interrupts and threads) or the "context\_switch" preset (only threads).
If using the "context\_switch" preset, Paraver might warn about not finding interrupt events, but it is safe to inspect the view.

Kernel Thread Ids
------------------

Uses Linux Kernel events to show the Thread id (TID) of the running thread in each core used by Nanos6.

This view requires the Linux Kernel "context\_switch" preset.

Kernel System Calls
------------------

Uses Linux Kernel events to show the system calls performed by each thread on cores used by Nanos6.
In this view, all threads but the traced application will be displayed as "Other threads" to enhance the readability of system calls.

This view requires the Linux Kernel "syscall" preset to trace all system calls.
If the user only wants to trace a set of syscalls, it can instead specify the "context\_switch" preset and provide a list of system call events manually using the `instrument.ctf.events.kernel.file` option in the Nanos6 configuration file.
