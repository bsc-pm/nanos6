/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "loader.h"
#include "function-interception.h"
#include "main-wrapper.h"
#include "api/nanos6/api-check.h"
#include "api/nanos6/bootstrap.h"
#include "api/nanos6/library-mode.h"
#include "api/nanos6/runtime-info.h"
#include "api/nanos6/taskwait.h"


#pragma GCC visibility push(default)

// Function interception is no longer triggered when loading the actual runtime
void nanos6_memory_allocation_interception_init();
// The following function is called so that the runtime can prepare for library unloading
void nanos6_memory_allocation_interception_fini();

#pragma GCC visibility pop


main_function_t *_nanos6_loader_wrapped_main = 0;


typedef struct {
	pthread_mutex_t _mutex;
	pthread_cond_t _cond;
	int _signaled;
} condition_variable_t;


typedef struct {
	int argc;
	char **argv;
	char **envp;
	int returnCode;
} main_task_args_block_t;


static void main_task_wrapper(void *argsBlock)
{
	main_task_args_block_t *realArgsBlock = (main_task_args_block_t *) argsBlock;
	
	assert(_nanos6_loader_wrapped_main != NULL);
	assert(realArgsBlock != NULL);
	
	realArgsBlock->returnCode = _nanos6_loader_wrapped_main(
		realArgsBlock->argc,
		realArgsBlock->argv,
		realArgsBlock->envp
	);
	
	char *reportPrefix = getenv("NANOS6_REPORT_PREFIX");
	if (reportPrefix != NULL) {
		for (void *it = nanos6_runtime_info_begin(); it != nanos6_runtime_info_end(); it = nanos6_runtime_info_advance(it)) {
			if (reportPrefix[0] != '\0') {
				printf("%s\t", reportPrefix);
			}
			
			nanos6_runtime_info_entry_t entry;
			nanos6_runtime_info_get(it, &entry);
			
			switch (entry.type) {
				case nanos6_integer_runtime_info_entry:
					printf("long\t");
					break;
				case nanos6_real_runtime_info_entry:
					printf("double\t");
					break;
				case nanos6_text_runtime_info_entry:
					printf("string\t");
					break;
			}
			
			printf("%s\t", entry.name);
			
			switch (entry.type) {
				case nanos6_integer_runtime_info_entry:
					printf("%li\t", entry.integer);
					break;
				case nanos6_real_runtime_info_entry:
					printf("%f\t", entry.real);
					break;
				case nanos6_text_runtime_info_entry:
					printf("%s\t", entry.text);
					break;
			}
			
			printf("%s\t%s\n", entry.units, entry.description);
		}
	}
}


static void main_completion_callback(void *args)
{
	condition_variable_t *condVar = (condition_variable_t *) args;
	assert(condVar != NULL);
	
	pthread_mutex_lock(&condVar->_mutex);
	condVar->_signaled = 1;
	pthread_cond_signal(&condVar->_cond);
	pthread_mutex_unlock(&condVar->_mutex);
}


int _nanos6_loader_main(int argc, char **argv, char **envp) {
	const nanos6_api_versions_t apiVersions = {
		.api_check_api_version = nanos6_api_check_api,
		
		.blocking_api_version = nanos6_blocking_api,
		.bootstrap_api_version = nanos6_bootstrap_api,
// 		.cuda_device_api_version = nanos6_cuda_device_api,
		.final_api_version = nanos6_final_api,
		.instantiation_api_version = nanos6_instantiation_api,
		.library_mode_api_version = nanos6_library_mode_api,
		.locking_api_version = nanos6_locking_api,
		.polling_api_version = nanos6_polling_api,
		.task_constraints_api_version = nanos6_task_constraints_api,
		.task_execution_api_version = nanos6_task_execution_api,
		.task_info_registration_api_version = nanos6_task_info_registration_api,
		.taskloop_api_version = nanos6_taskloop_api,
		.taskwait_api_version = nanos6_taskwait_api,
		.utils_api_version = nanos6_utils_api
	};
	
	if (nanos6_check_api_versions(&apiVersions) != 1) {
		fprintf(stderr, "Error: this executable was compiled for a different Nanos6 version. Please recompile and link it.\n");
		return 1;
	}
	
	nanos6_memory_allocation_interception_init();
	
	if (_nanos6_exit_with_error) {
		return _nanos6_exit_with_error;
	}
	
	// First half of the initialization
	nanos_preinit();
	if (_nanos6_exit_with_error) {
		return _nanos6_exit_with_error;
	}
	
	condition_variable_t condVar = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
	
	// Spawn the main task
	main_task_args_block_t argsBlock = { argc, argv, envp, 0 };
	nanos_spawn_function(main_task_wrapper, &argsBlock, main_completion_callback, &condVar, "main");
	
	if (_nanos6_exit_with_error) {
		return _nanos6_exit_with_error;
	}
	
	// Second half of the initialization
	nanos_init();
	
	// Wait for the completion callback
	pthread_mutex_lock(&condVar._mutex);
	while (condVar._signaled == 0) {
		pthread_cond_wait(&condVar._cond, &condVar._mutex);
	}
	pthread_mutex_unlock(&condVar._mutex);
	
	// Terminate
	nanos_shutdown();
	
	nanos6_memory_allocation_interception_fini();
	
	return argsBlock.returnCode;
}

