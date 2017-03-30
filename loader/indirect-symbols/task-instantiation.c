#include "resolve.h"


void nanos_create_task(
	nanos_task_info *task_info,
	nanos_task_invocation_info *task_invocation_info,
	size_t args_block_size,
	/* OUT */ void **args_block_pointer,
	/* OUT */ void **task_pointer,
	size_t flags
) {
	typedef void nanos_create_task_t(
		nanos_task_info *task_info,
		nanos_task_invocation_info *task_invocation_info,
		size_t args_block_size,
		/* OUT */ void **args_block_pointer,
		/* OUT */ void **task_pointer,
		size_t flags
	);
	
	static nanos_create_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_create_task_t *) _nanos6_resolve_symbol("nanos_create_task", "essential", NULL);
	}
	
	(*symbol)(task_info, task_invocation_info, args_block_size, args_block_pointer, task_pointer, flags);
}


void nanos_submit_task(void *task)
{
	typedef void nanos_submit_task_t(void *task);
	
	static nanos_submit_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_submit_task_t *) _nanos6_resolve_symbol("nanos_submit_task", "essential", NULL);
	}
	
	(*symbol)(task);
}


void nanos_spawn_function(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label)
{
	typedef void nanos_spawn_function_t(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label);
	
	static nanos_spawn_function_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_spawn_function_t *) _nanos6_resolve_symbol("nanos_spawn_function", "essential", NULL);
	}
	
	(*symbol)(function, args, completion_callback, completion_args, label);
}


