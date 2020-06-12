# Runtime Throttle

For programs that will generate in the order of millions of tasks, specially those that will run for a long time, and rely only in data dependencies to provide synchronization (which is the nature of the OmpSs-2 model), it is a possiblity that the runtime will run out of memory because all the data structures for the tasks and dependencies will be created in a short amount of time.

To prevent this issue, the runtime includes a "throttle" mechanism which can monitor the memory usage and stop creating tasks while there is high memory pressure, without incurring too much overhead because the stopped threads will be reused executing ready tasks. The main idea is that with this mechanism active, there is no way the runtime can exceed the memory budget during execution, and the execution time will be similar (if not better) than in a system with infinite memory.

## Requisites

The throttle mechanism will only be included in the runtime if it is configured with a valid jemalloc installation, through the `--with-jemalloc` configure flag.

## Usage

This feature is controlled through the following environment variables

<table><tbody><tr><td> <strong>Environment Variable</strong> </td><td><strong>Description</strong></td><td> <strong>Default Value</strong>
</td></tr><tr><td> <em>NANOS6_THROTTLE</em> </td><td> Boolean variable that enables the throttle mechanism </td><td><em>false</em>
</td></tr><tr><td> <em>NANOS6_THROTTLE_TASKS</em> </td><td> Maximum absolute number of alive childs that any task can have. It is divided by 10 at each nesting level </td><td><em>5000000</em>
</td></tr><tr><td> <em>NANOS6_THROTTLE_PRESSURE</em> </td><td> Percentage of memory budget used at which point the number of tasks allowed to exist will be decreased linearly until reaching 1 at 100% memory pressure </td><td><em>70</em>
</td></tr><tr><td> <em>NANOS6_THROTTLE_MAX_MEMORY</em> </td><td> Maximum used memory (memory budget) </td><td><em> Available Physical Memory / 2</em>
</td></tr></tbody></table>

Note that `NANOS6_THROTTLE_MAX_MEMORY` can be set in terms of bytes or in memory units. For example: `export NANOS6_THROTTLE_MAX_MEMORY=50GB`.