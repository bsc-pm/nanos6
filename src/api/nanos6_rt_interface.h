#ifndef NANOS6_RT_INTERFACE_H
#define NANOS6_RT_INTERFACE_H

#include <stddef.h>

#define __NANOS6__

#ifdef __cplusplus
extern "C" {
#endif


//! \brief Struct that contains the common parts that all tasks of the same type share
typedef struct
{
	//! \brief Wrapper around the actual task implementation
	//! 
	//! \param[in,out] args_block A pointer to a block of data for the parameters
	void (*run)(void *args_block);
	
	//! \brief Function that the runtime calls to retrieve the information needed to calculate the dependencies
	//! 
	//! This function should call the nanos_register_input_dep, nanos_register_output_dep and nanos_register_inout_dep
	//! functions to pass to the runtime the information needed to calculate the dependencies
	//! 
	//! \param[in] handler a handler to be passed on to the registration functions
	//! \param[in] args_block a pointer to a block of data for the parameters partially initialized
	void (*register_depinfo)(void *handler, void *args_block);
	
	//! \brief Function that the runtime calls to retrieve the information needed to perform the data copies between devices
	//! 
	//! This function should call the nanos_register_copy_in, nanos_register_copy_out and nanos_register_copy_inout
	//! functions to pass to the runtime the information needed to perform the data copies between devices
	//! 
	//! \param[in] handler a handler to be passed on to the registration functions
	//! \param[in] args_block a pointer to a block of data for the parameters partially initialized
	void (*register_copies)(void *handler, void *args_block);
	
	//! \brief A string that identifies the type of task
	char const *task_label;
	
	//! \brief A string that identifies the source location of the definition of the task
	char const *declaration_source;
} nanos_task_info __attribute__((aligned(64)));


//! \brief Struct that contains data shared by all tasks invoked at fixed location in the source code
typedef struct
{
	//! \brief A string that identifies the source code location of the task invocation
	char const *invocation_source;
} nanos_task_invocation_info __attribute__((aligned(64)));


//! \brief Allocate space for a task and its parameters
//! 
//! This function creates a task and allocates space for its parameters.
//! After calling it, the user code should fill out the block of data stored in args_block_pointer,
//! and call nanos_submit_task with the contents stored in task_pointer.
//! 
//! \param[in] task_info a pointer to the nanos_task_info structure
//! \param[in] task_invocation_info a pointer to the nanos_task_invocation_info structure
//! \param[in] args_block_size size needed to store the paramerters passed to the task call
//! \param[out] args_block_pointer a pointer to a location to store the pointer to the block of data that will contain the parameters of the task call
//! \param[out] task_pointer a pointer to a location to store the task handler
void nanos_create_task(
	nanos_task_info *task_info,
	nanos_task_invocation_info *task_invocation_info,
	size_t args_block_size,
	/* OUT */ void **args_block_pointer,
	/* OUT */ void **task_pointer
);


//! \brief Submit a task
//! 
//! This function should be called after filling out the block of parameters of the task. See nanos_create_task.
//! 
//! \param[in] task The task handler
void nanos_submit_task(void *task);


//! \brief Block the control flow of the current task until all of its children have finished
//!
//! \param[in] invocation_source A string that identifies the source code location of the invocation
void nanos_taskwait(char const *invocation_source);


//! \brief Register a task read access on linear range of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_read_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task write access on linear range of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_write_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task read and write access on linear range of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length);


//! \brief User-side lock primitive with initialization on first call
//!
//! Performs an user-side lock over a mutex (of type void *) that must be initially
//! initialized to nullptr. The first call to this function performs the actual
//! mutex allocation and stores the handler in the address that is passed.
//!
//! \param[in,out] handlerPointer a pointer to the handler, which is of type void *, that represent the mutex
//! \param[in] invocation_source A string that identifies the location of the critical region in the source code
void nanos_user_lock(void **handlerPointer, char const *invocation_source);

//! \brief User-side unlock primitive
//!
//! Performs an user-side unlock over a mutex (of type void *) initialized during
//! the first call to nanos_user_lock.
//!
//! \param[in] handlerPointer a pointer to the handler, which is of type void *, that represent the mutex
void nanos_user_unlock(void **handlerPointer);


// TODO: nanos_register_input_copy, nanos_register_output_copy and nanos_register_inout_copy


#ifdef __cplusplus
}
#endif

#endif // NANOS6_RT_INTERFACE_H
