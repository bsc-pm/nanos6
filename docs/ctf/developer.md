
NOTE: this section is under construction

API calls
=========

The CTF backend provides tracepoints for Nanos6 API calls with the objective of identifying periods of time spend running task code and runtime code. Also, these Nanos6 entry and exit points are used to carry hardware counters (HWC) information. On Nanos6 entry, HWC associated with a task are issued. On exit, Nanos6 HWC are issued.

The entry and exit points also need to deal with external threads calling API services and even the own runtime code calling its own API. As a result, the entry/exit tracepoints are a bit complex.

Nanos6 calling its own API
--------------------------

When Nanos6 calls its own API, it means that the API call comes from runtime code instead of task code. This has two implications:

 1 - We might not really need to emit these tracepoints because they are of no interest (we might be only interested on the transition between task and runtime to keep track of how much time is spend in runtime and task context)
 2 - If we want to emit this tracepoints, there is no need to issue HWC information. The entry information It has already been issued the moment the task to runtime transition was made and the exit information will be made when returning to task context.

Point 1 is easily achieved. But point 2 it is not. Once an even has been defined including context information, this context must be always emitted. Hence, it is not possible to avoid emitting it.

Can we apply a similar technique to External Threads to allow worker thread emit events with or without HWC information? External threads do never emit HWC information. Events emitted by external threads belong to another type of stream, and such events are defined without HWC context. We cannot easily apply the same technique here because the stream an event belongs to is associated with the per-cpu circular buffer i.e. external threads have a virtual CPU with its own buffer and it's own stream, which does not include HWC definitions. Similarly, the event emitted by a worker thread are bound to the stream associated with it's per-cpu buffer, but this stream event's definitions are defined including HWC information. Enabling a worker thread to sometimes emit HWC information and sometimes, not would require to have two circular buffers per CPU, which excessively increases the design complexity.

An option would be to just emit always HWC information, even when an API is called from Nanos6 context and it is not strictly needed. However, this would require to perform more hardware counter register reads than necessary, and we would need to distinguish at each entry and exit points whether we need to update task counters or runtime counters. Also, the current CTF design does not allow a tracepoint to emit runtime HWC information or Task HWC information, it must always emit one or the other.

Another option is to always emit HWC information, but with invalid values. This would be confusing for the user and would further increase the CTF trace size.

Even another option is to decouple HWC information form events. This means that events would no longer have a HWC context, but we would have an exclusive tracepoint used to emit HWC information only. This is the most flexible option to dyanmically decide whether to emit HWC information at one point or not, but this would increase even more the CTF trace size, given that each of this tracepoints would have its own timestamp and event id. Instead, service HWC information into other event's context allows us to "reuse" an already existing event.

The easiest solution (and the one implemented) is to duplicate the event's definition for the worker thread stream but without HWC information. Hence, for the `nanos6_create_task()` API there will be two events associated with the entry point: `nanos6:tc:task_create_entry` and `nanos6:oc:task_create_entry`. Notice the "tc" and "oc" acronyms whose meaning is "Task Context" and "Other Context" respectively. Tracepoints with "tc" mark a transition between task context and runtime context (in either direction). These tracepoints contains HWC information. Tracepoints with "oc" can be either called from runtime context or external thread context.

ctf2prv benefits
----------------

Additionally, having this distinction between task context and other context eases the ctf2prv conversion. In ctf2prv there are two modes to see tasks:

 1 - Continuous view: Tasks are drawn in paraver uninterruptedly unless they block or end. Example views are "Tasks", "Task Source" or "Task Id" views.
 2 - Fragmented view: Additionally to the continuous view, it is also displayed the time a task spends in runtime mode while calling a Nanos6 API. Example views are "Runtime Status Simple" and "Runtime Subsystems".

The "Continuous View" mode ignores the Nanos6 entry/exit tracepoints. Instead, the "Fragmented View" depends on them. The "Runtime Subystems" view is interested in drawing all entry/exit points, because it displays what the runtime is doing at all times i.e. it draws entry/exit tracepoints regardless of them being called by an external thread, a worker thread coming from task context or a worker thread coming from runtime context (this is "tc" and "oc" tracepoints). The "Runtime Status Simple" is only interested in task and runtime transitions i.e. entry/exit points called by worker threads when transitioning between task and runtime context. This is accomplished by only using the "tc" tracepoints.

The ctf2prv converter could still create such views without considering "tc" and "oc" distinction by keeping a stack of events and avoid drawing entry/exit tracepoints if a particular core is already in runtime state, but having this distinction (necessary because of HWC information) makes the processes simpler.

Summary
-------

 * At each Nanos6 entry point (such as `nanos6_create_task()`) two tracepoints are emitted, one for entry and one for exit.
 * Some of this entry and exit tracepoints have two variations "tc" (Task Context) and "oc" (Other Context). For example: `nanos6:tc:create_task_entry` and `nanos6:oc:spawn_function_exit`. The "tc" variant marks a transition between task and runtime context. The "oc" variant marks an API called either from runtime context (the runtime calls its own API) or external thread context (an external thread calls this API).
 * Only the "tc" variants might carry HWC information (if HWC are enabled). The entry tracepoint carries task HWC information and the exit tracepoint carries runtime HWC information.
 * Whether a Nanos6 entry point has this two variations or not, depends on two requirements:
   1 - Will the runtime call the API from within the runtime context? If so, wee need the "oc" variant to avoid emitting HWC information, or we could simply not emit this tracepoint.
   2 - Do we want to see this tracepoint in Paraver? Some internal API calls are of no interest and we do not need to see them. Also, having less tracepoints, means having a smaller CTF trace. If we want to see the event in all cases, then we need the "oc" variant.
   For instance, `nanos6_block_current_task()` will always be called from a task context (because it blocks the current task) and hence it will not be called from runtime context. Therefore, its entry/exit tracepoints do not need a "oc" version. However, the `nanos6_unblock_task()` function might be called from runtime context, and we want to see them in Paraver to identify External threads unblocking tasks. Hence, we need the "oc" variant.
