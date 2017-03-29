#include "resolve.h"


void *nanos_get_current_blocking_context()
{
	typedef void *nanos_get_current_blocking_context_t();
	
	static nanos_get_current_blocking_context_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_current_blocking_context_t *) _nanos6_resolve_symbol("nanos_get_current_blocking_context", "task blocking", NULL);
	}
	
	return (*symbol)();
}


void nanos_block_current_task(void *blocking_context)
{
	typedef void nanos_block_current_task_t(void *);
	
	static nanos_block_current_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_block_current_task_t *) _nanos6_resolve_symbol("nanos_block_current_task", "task blocking", NULL);
	}
	
	(*symbol)(blocking_context);
}


void nanos_unblock_task(void *blocking_context)
{
	typedef void nanos_unblock_task_t(void *blocked_task_handler);
	
	static nanos_unblock_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_unblock_task_t *) _nanos6_resolve_symbol("nanos_unblock_task", "task blocking", NULL);
	}
	
	(*symbol)(blocking_context);
}

