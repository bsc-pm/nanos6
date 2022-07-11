/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_TASK_INSTANTIATION_H
#define NANOS6_TASK_INSTANTIATION_H

#include <stddef.h>
#include <unistd.h>

#include "major.h"

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Data type to express priorities
typedef signed long nanos6_priority_t;


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_task_execution_api
//! \brief This needs to be incremented every time there is an update to the nanos6_task_info::run
enum nanos6_task_execution_api_t { nanos6_task_execution_api = 1 };


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_task_constraints_api
//! \brief This needs to be incremented every time there is an update to nanos6_task_constraints_t
enum nanos6_task_constraints_api_t { nanos6_task_constraints_api = 1 };


typedef struct
{
	size_t cost;
} nanos6_task_constraints_t;


typedef enum {
	nanos6_host_device = 0,
	nanos6_cuda_device,
	nanos6_openacc_device,
	nanos6_cluster_device,
	nanos6_opencl_device,
	nanos6_fpga_device,
	nanos6_device_type_num = 6
} nanos6_device_t;

typedef struct {
	size_t sizes[6];
	size_t shm_size;
} nanos6_device_info_t;


typedef struct
{
	size_t local_address;
	size_t device_address;
} nanos6_address_translation_entry_t;


typedef struct
{
	//! \brief Runtime device identifier (original type nanos6_device_t)
	int device_type_id;

	//! \brief Wrapper around the actual task implementation
	//!
	//! \param[in,out] args_block A pointer to a block of data for the parameters
	//! \param[in] device_env a pointer to device-specific data
	//! \param[in] address_translation_table one entry per task symbol that maps host addresses to device addresses
	void (*run)(void *args_block, void *device_env, nanos6_address_translation_entry_t *address_translation_table);

	//! \brief Function to retrieve constraint information about the task (cost, memory requirements, ...)
	//!
	//! \param[in] args_block A pointer to a block of data for the parameters
	//! \param[in,out] constraints A pointer to the struct where constraints are written
	void (*get_constraints)(void *args_block, nanos6_task_constraints_t *constraints);

	//! \brief A string that identifies the type of task
	char const *task_type_label;

	//! \brief A string that identifies the source location of the definition of the task
	char const *declaration_source;

	//! \brief In device tasks, name of the kernel to be loaded and executed
	const char *device_function_name;
} nanos6_task_implementation_info_t;


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_task_info_contents
//! \brief This needs to be incremented every time that there is a change in nanos6_task_info
enum nanos6_task_info_contents_t { nanos6_task_info_contents = 4 };

//! \brief Struct that contains the common parts that all tasks of the same type share
typedef struct __attribute__((aligned(64))) {
	//! \brief Number of symbols accessible by the task
	int /*const*/ num_symbols; // TODO: removed const for obstructing construction, until further decision

	//! \brief Function that the runtime calls to retrieve the information needed to calculate the dependencies
	//!
	//! This function should call the nanos6_register_input_dep, nanos6_register_output_dep and nanos6_register_inout_dep
	//! functions to pass to the runtime the information needed to calculate the dependencies
	//!
	//! \param[in] args_block a pointer to a block of data for the parameters partially initialized
	//! \param[in] taskloop_bounds a pointer to the bounds of a taskloop, if the task is taskloop
	//! \param[in] handler a handler to be passed on to the registration functions
	void (*register_depinfo)(void *args_block, void *taskloop_bounds, void *handler);

	//! \brief Function that the runtime calls to run the onready action
	//!
	//! This function should be called by the runtime after the dependencies of the task have been satisfied. This function
	//! may register external events to delay the execution of the task, but cannot block the current thread in any blocking
	//! operation (e.g., taskwait or explicit blocking)
	//!
	//! \param[in] args_block a pointer to a block of data for the parameters partially initialized
	void (*onready_action)(void *args_block);

	//! \brief Function that the runtime calls to obtain a user-specified priority for the task instance
	//!
	//! Note that this field can be null to indicate the default priority.
	//!
	//! \param[in] args_block a pointer to a block of data for the parameters partially initialized
	//! \param[out] priority a pointer to a block of data where the desired priority is stored
	void (*get_priority)(void *args_block, nanos6_priority_t *priority);

	//! \brief Number of implementations of the task
	int /*const*/ implementation_count; // TODO: removed const for obstructing construction, until further decision

	//! \brief Array of implementations
	nanos6_task_implementation_info_t /*const*/ *implementations; // TODO: removed const for obstructing construction, until further decision

	//! \brief Function that the runtime calls to perform any cleanup needed in the block of data of the parameters
	//!
	//! \param[in,out] args_block A pointer to a block of data for the parameters
	void (*destroy_args_block)(void *args_block);

	//! \brief Function that the runtime calls to perform a copy of the block of data of the parameters
	//!
	//! \param[in] src_args_block A pointer to the source block of parameters to be copied
	//! \param[in,out] dest_args_block_pointer A pointer to a location to store the pointer to the destination array where the block of parameters is to be copied. If the task was created with the nanos6_preallocated_args_block flag, then it will be initialized by this function, otherwise, the runtime will initialize it
	void (*duplicate_args_block)(const void *src_args_block, void **dest_args_block_pointer);

	//! \brief Array of functions that the runtime calls to initialize task
	//! reductions' private storage
	//!
	//! \param[out] oss_priv a pointer to the data to be initialized
	//! \param[in] oss_orig a pointer to the original data, which may be used
	//! \param[in] size the (in Bytes) size of the data to be initialized
	//! during initialization
	void (**reduction_initializers)(void *oss_priv, void *oss_orig, size_t size);

	//! \brief Array of functions that the runtime calls to combine task
	//! reductions' private storage
	//!
	//! \param[out] oss_out a pointer to the data where the combination is to
	//! be performed
	//! \param[in] oss_in a pointer to the data which needs to be combined
	//! \param[in] size the size (in Bytes) of the data to be combined
	void (**reduction_combiners)(void *oss_out, void *oss_in, size_t size);

	//! \brief A pointer to data structures related to this type of task
	void *task_type_data;

	//! \brief Number of task arguments
	int num_args;

	//! \brief Table of sizeofs for all num_args arguments of the task
	int *sizeof_table;

	//! \brief Table of offsets for all num_args arguments of the task
	int *offset_table;

	//! \brief list of indexes to map a symbol  into arg
	int *arg_idx_table;

} nanos6_task_info_t;


//! \brief Struct that contains data shared by all tasks invoked at fixed location in the source code
typedef struct __attribute__((aligned(64))) {
	//! \brief A string that identifies the source code location of the task invocation
	char const *invocation_source;
} nanos6_task_invocation_info_t;


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_instantiation_api
//! \brief This needs to be incremented on every change to the instantiation API
enum nanos6_instantiation_api_t { nanos6_instantiation_api = 6 };

typedef enum {
	//! Specifies that the task will be a final task
	nanos6_final_task = (1 << 0),
	//! Specifies that the task is in "if(0)" mode
	nanos6_if_0_task = (1 << 1),
	//! Specifies that the task is really a taskloop
	nanos6_taskloop_task = (1 << 2),
	//! Specifies that the task is really a taskfor
	nanos6_taskfor_task = (1 << 3),
	//! Specifies that the task has the "wait" clause
	nanos6_waiting_task = (1 << 4),
	//! Specifies that the args_block is preallocated from user side
	nanos6_preallocated_args_block = (1 << 5),
	//! Specifies that the task has been verified by the user, hence it doesn't need runtime linting
	nanos6_verified_task = (1 << 6),
	//! Specifies that the task is really a taskiter
	nanos6_taskiter_task = (1 << 7),
	//! Specifies that the task has the "update" clause
	nanos6_update_task = (1 << 8)
} nanos6_task_flag_t;


//! \brief Allocate space for a task and its parameters
//!
//! This function creates a task and allocates space for its parameters.
//! After calling it, the user code should fill out the block of data stored in args_block_pointer,
//! and call nanos6_submit_task with the contents stored in task_pointer.
//!
//! \param[in] task_info a pointer to the nanos6_task_info_t structure
//! \param[in] task_invocation_info a pointer to the nanos6_task_invocation_info_t structure
//! \param[in] task_label a string that identifies the task
//! \param[in] args_block_size size needed to store the parameters passed to the task call
//! \param[in,out] args_block_pointer a pointer to a location to store the pointer to the block of data that will contain the parameters of the task call. Input if flags contains nanos6_preallocated_args_block, out otherwise
//! \param[out] task_pointer a pointer to a location to store the task handler
//! \param[in] flags the flags of the task
//! \param[in] num_deps the expected number of dependencies of this task or -1 if undefined
void nanos6_create_task(
	nanos6_task_info_t *task_info,
	nanos6_task_invocation_info_t *task_invocation_info,
	char const *task_label,
	size_t args_block_size,
	/* OUT */ void **args_block_pointer,
	/* OUT */ void **task_pointer,
	size_t flags,
	size_t num_deps);


//! \brief Submit a task
//!
//! This function should be called after filling out the block of parameters of the task. See nanos6_create_task.
//!
//! \param[in] task The task handler
void nanos6_submit_task(void *task);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_TASK_INSTANTIATION_H */
