/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos6_get_current_event_counter(void)
{
	typedef void *nanos6_get_current_event_counter_t(void);
	
	static nanos6_get_current_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_current_event_counter_t *) _nanos6_resolve_symbol("nanos6_get_current_event_counter", "essential", NULL);
	}
	
	return (*symbol)();
}

void nanos6_increase_current_task_event_counter(void *event_counter, unsigned int increment)
{
	typedef void nanos6_increase_current_task_event_counter_t(void *event_counter, unsigned int increment);
	
	static nanos6_increase_current_task_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_increase_current_task_event_counter_t *) _nanos6_resolve_symbol("nanos6_increase_current_task_event_counter", "essential", NULL);
	}
	
	(*symbol)(event_counter, increment);
}

void nanos6_decrease_task_event_counter(void *event_counter, unsigned int decrement)
{
	typedef void nanos6_decrease_task_event_counter_t(void *event_counter, unsigned int decrement);
	
	static nanos6_decrease_task_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_decrease_task_event_counter_t *) _nanos6_resolve_symbol("nanos6_decrease_task_event_counter", "essential", NULL);
	}
	
	(*symbol)(event_counter, decrement);
}

#pragma GCC visibility pop

