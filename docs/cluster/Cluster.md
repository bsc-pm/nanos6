# OmpSs-2@Cluster programming model

OmpSs-2@Cluster is an extension of the OmpSs-2 programming model which allows programmers to write OmpSs-2 applications that can run on top of multiple
distributed memory cluster nodes. Essentially, OmpSs-2@Cluster introduces more strict data-flow semantics over traditional OmpSs-2 and an extended API
library API which provides memory allocation functions for allocating memory that Nanos6 is handling.

## Memory model

OmpSs-2@Cluster introduces a memory model that defines two different memory types in which distributed computations can take place, *local* and *distributed*
memory.

Local memory is cluster-capable memory that is allocated on a single node and can be dereferenced directly within the context of the task that allocated it.
Nanos6 uses local memory in for the stack frames of the worker threads executing tasks. Moreover, user applications can allocate local memory through the
following API:

```c
//! Allocate local memory
//!
//! \param[in] size is the size (in bytes) of local memory to allocate
//!
//! \returns a pointer to local memory
void *nanos6_lmalloc(size_t size);

//! Deallocate a local array
//!
//! \param[in] ptr is a pointer to memory previously allocated with nanos6_lmalloc
//! \param[in] size is the size of the local memory allocation
void nanos6_lfree(void *ptr, size_t size);
```

and example of valid usage of local memory:

```c

...

int *local_array = nanos6_lmalloc(1024 * sizeof(int));

// We can dereference local memory after allocating it
for (size_t i = 0; i < 1024; ++i) {
	local_array[i] = init_element(i);
}

// Local memory can participate in cluster-computations, i.e. used by subtasks
#pragma oss task inout(array[i;1024])
foo(local_array, 1024);

// We still need a taskwait before accessing it again within this task
#pragma oss taskwait

printf("%d\n", local_array[0]);

nanos6_lfree(local_array, 1024 * sizeof(int));

```

Distributed memory, is cluster-capable memory that is allocated collectively across all the nodes that participate in the execution and can be dereferenced
only through sub-tasks of the task that allocated it. When allocating distributed memory the user application can define a *distribution policy*. This policy
is meant to pass to the runtime hints regarding the pattern with which the application will access allocated buffer. The allocation API is the following:

```c
/* Distributed memory API */

//! \brief Allocate distributed memory
//!
//! Distributed memory is a clsuter type of memory that can only be
//! accessed from within a task.
//!
//! \param[in] size is the size (in bytes) of distributed memory to allocate
//! \param[in] policy is the data distribution policy based on which we will
//!            distribute data across cluster nodes
//! \param[in] num_dimensions is the number of dimensions across which the data
//!            will be distributed
//! \param[in] dimensions is an array of num_dimensions elements, which contains
//!            the size of every distribution dimension
//!
//! \returns a pointer to distributed memory
void *nanos6_dmalloc(size_t size, nanos6_data_distribution_t policy, size_t num_dimensions, size_t *dimensions);

//! \brief deallocate a distributed array
//
//! \param[in] ptr is a pointer to memory previously allocated with nanos6_dmalloc
//! \param[in] size is the size of the distributed memory allocation
void nanos6_dfree(void *ptr, size_t size);
```

At the moment, we only define one distribution policy: `nanos6_equpart_distribution` which distributes equally the allocated memory region across all cluster
nodes. In future releases, we plan to add additional distribution policies, matching with user application access patterns.

Here is a  example of how distributed memory should be allocated and used:

```c
int *distributed_array =
	nanos6_dmalloc(1024 * sizeof(int), nanos6_equpart_distribution,
		// These arguments are not used by the nanos6_equpart_distribution
		0, NULL);

/* We can *NOT* dereference distributed memory after allocating it
 * for (size_t i = 0; i < 1024; ++i) {
 *	distributed_array[i] = init_element(i);
 * }
 */

// We can only access it through sub-tasks
for (size_t i = 0; i < 1024; ++i) {
	#pragma oss task out(distributed_array[i])
	distributed_array[i] = init_element(i);
}

#pragma oss task inout(array[i;1024])
foo(distributed_array, 1024);

/* We do *NOT* need a taskwait before accessing it again within this task
 * since, anyway we can only access it through subtasks
 *
 * #pragma oss taskwait
 */

#pragma oss task in(distributed_array[0]);
printf("%d\n", distributed_array[0]);

/* But we *DO* need a taskwait before deallocating the memory, to make sure
 * all subtasks using it have completed */
#pragma oss taskwait

nanos6_dfree(distributed_array, 1024 * sizeof(int));
```

Nanos6 allows users to manage the size of the local and distributed memory regions, by means of setting environment variables.
The two environment variables of interest are `NANOS6_LOCAL_MEMORY` and `NANOS6_DISTRIBUTED_MEMORY`, which can be set to size values like:

```sh
export NANOS6_LOCAL_MEMORY=2GB
export NANOS6_DISTRIBUTED_MEMORY=120GB
```

Notice: Nanos6 will allocate the sum of these variables on all instances of the runtime in a clustered execution. This might require the
user to set accordingly the *overcommit* settings of the platform. Take a look at the [System Requirements](#system-requirements) section.

## Data-flow semantics

OmpSs-2@Cluster supports the *in*, *out*, *inout* dependency clauses and their *weak* equivalents. The difference between the Cluster and the shared memory version of OmpSs-2
lies in the fact that the former **requires** that all the memory accesses of a task appear in its dependency list. With OmpSs-2@Cluster it is not enough that the programmer
defines a subset of the accesses as a dependencies, so as to just enforce the correct ordering of the task. If the programmer fails to annotate her program correctly in that
sense the results of the computation might be incorrect.

These restrictions, in addition with the requirements for accessing [distributed memory](#memory-model) are essentially the additional things that an OmpSs-2 programmer needs
to take into account when writing applications for OmpSs-2@Cluster. This means, essentially, that any correct OmpSs-2@Cluster program is a correct shared-memory OmpSs-2 program
as well, but not vice versa.

## Installation

### Build requirements

To install Nanos6 with support for OmpSs-2@Clusters in addition to the prerequisites of the base Nanos6 runtime which you can find [here](../../README.md),
you will need a MPI library with support for multithreading, i.e. `MPI_THREAD_MULTIPLE`, enabled.

### Build procedure

Follow the same steps as in [README.md](../../README.md). During configuration add the `--enable-cluster` flag:

```sh
./configure --prefix=INSTALLATION_PREFIX ...other options... --enable-cluster ...other options...
make
make install
```


## Execution

### Preparing the environment

Apart from enabling cluster support in Nanos6 during compilation, users of OmpSs-2@Cluster need to enable it at runtime by setting the `NANOS6_COMMUNICATION`
environment variable. This variable determines which communication layer will be used by Nanos6 for Cluster communication. At the moment, we only support
2-sided MPI implementations, so:

```sh
export NANOS6_COMMUNICATION=mpi-2sided
```

If this variable is not set, the application will run as if cluster is disabled.

### Launching the application

You launch an OmpSs-2@Cluster application using the standard utility provided by the MPI library you used to build Nanos6 with Cluster support. For example,

```sh
export NANOS6_COMMUNICATION=mpi-2sided
mpirun -np 16 taskset -c 0-47 ./your_ompss_cluster_app args...
```

If you are running on a system with a batch scheduler, such as SLURM, follow the instructions of your cluster for submitting distributed memory jobs on the
scheduler. For example, a SLURM job script could look like:

```sh
#!/bin/bash

#SBATCH --job-name=my_ompss_cluster_app
#SBATCH --time=30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=48


export NANOS6_COMMUNICATION=mpi-2sided
srun ./your_ompss_cluster_app args...
```

### System requirements

At the moment, cluster execution requires the system to have disabled the address randomization feature. Moreover, depending on the total address space the
application needs to manage, i.e. `NANOS6_LOCAL_MEMORY` and `NANOS6_DISTRIBUTED_MEMORY` you might need to change the memory overcommit feature of your kernel.

These configurations can be done on a Linux system like this:

```sh
# Disable address randomization
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Always allow memory overcommit
echo 1 | sudo tee /proc/sys/vm/overcommit_memory
```
