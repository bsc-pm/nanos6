/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos_get_current_event_counter()
{
	typedef void *nanos_get_current_event_counter_t();
	
	static nanos_get_current_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_current_event_counter_t *) _nanos6_resolve_symbol("nanos_get_current_event_counter", "essential", NULL);
	}
	
	return (*symbol)();
}

void nanos_increase_current_task_event_counter(void *event_counter, unsigned int increment)
{
	typedef void nanos_increase_current_task_event_counter_t(void *event_counter, unsigned int increment);
	
	static nanos_increase_current_task_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_increase_current_task_event_counter_t *) _nanos6_resolve_symbol("nanos_increase_current_task_event_counter", "essential", NULL);
	}
	
	(*symbol)(event_counter, increment);
}

void nanos_decrease_task_event_counter(void *event_counter, unsigned int decrement)
{
	typedef void nanos_decrease_task_event_counter_t(void *event_counter, unsigned int decrement);
	
	static nanos_decrease_task_event_counter_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_decrease_task_event_counter_t *) _nanos6_resolve_symbol("nanos_decrease_task_event_counter", "essential", NULL);
	}
	
	(*symbol)(event_counter, decrement);
}

#pragma GCC visibility pop

