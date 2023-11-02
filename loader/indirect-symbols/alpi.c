/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/alpi.h"

#pragma GCC visibility push(default)

const char *alpi_error_string(int error)
{
	typedef const char *alpi_error_string_t(int);

	static alpi_error_string_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_error_string_t *) _nanos6_resolve_symbol("alpi_error_string", "alpi", NULL);
	}

	return (*symbol)(error);
}

int alpi_version_check(int major, int minor)
{
	typedef int alpi_version_check_t(int, int);

	static alpi_version_check_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_version_check_t *) _nanos6_resolve_symbol("alpi_version_check", "alpi", NULL);
	}

	return (*symbol)(major, minor);
}

int alpi_version_get(int *major, int *minor)
{
	typedef int alpi_version_get_t(int *, int *);

	static alpi_version_get_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_version_get_t *) _nanos6_resolve_symbol("alpi_version_get", "alpi", NULL);
	}

	return (*symbol)(major, minor);
}

int alpi_task_self(struct alpi_task **task)
{
	typedef int alpi_task_self_t(struct alpi_task **);

	static alpi_task_self_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_self_t *) _nanos6_resolve_symbol("alpi_task_self", "alpi", NULL);
	}

	return (*symbol)(task);
}

int alpi_task_block(struct alpi_task *task)
{
	typedef int alpi_task_block_t(struct alpi_task *);

	static alpi_task_block_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_block_t *) _nanos6_resolve_symbol("alpi_task_block", "alpi", NULL);
	}

	return (*symbol)(task);
}

int alpi_task_unblock(struct alpi_task *task)
{
	typedef int alpi_task_unblock_t(struct alpi_task *);

	static alpi_task_unblock_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_unblock_t *) _nanos6_resolve_symbol("alpi_task_unblock", "alpi", NULL);
	}

	return (*symbol)(task);
}

int alpi_task_events_increase(struct alpi_task *task, uint64_t increment)
{
	typedef int alpi_task_events_increase_t(struct alpi_task *, uint64_t);

	static alpi_task_events_increase_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_events_increase_t *) _nanos6_resolve_symbol("alpi_task_events_increase", "alpi", NULL);
	}

	return (*symbol)(task, increment);
}

int alpi_task_events_decrease(struct alpi_task *task, uint64_t decrement)
{
	typedef int alpi_task_events_decrease_t(struct alpi_task *, uint64_t);

	static alpi_task_events_decrease_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_events_decrease_t *) _nanos6_resolve_symbol("alpi_task_events_decrease", "alpi", NULL);
	}

	return (*symbol)(task, decrement);
}

int alpi_task_waitfor_ns(uint64_t target_ns, uint64_t *actual_ns)
{
	typedef int alpi_task_waitfor_ns_t(uint64_t, uint64_t *);

	static alpi_task_waitfor_ns_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_waitfor_ns_t *) _nanos6_resolve_symbol("alpi_task_waitfor_ns", "alpi", NULL);
	}

	return (*symbol)(target_ns, actual_ns);
}

int alpi_attr_create(struct alpi_attr **attr)
{
	typedef int alpi_attr_create_t(struct alpi_attr **);

	static alpi_attr_create_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_attr_create_t *) _nanos6_resolve_symbol("alpi_attr_create", "alpi", NULL);
	}

	return (*symbol)(attr);
}

int alpi_attr_destroy(struct alpi_attr *attr)
{
	typedef int alpi_attr_destroy_t(struct alpi_attr *);

	static alpi_attr_destroy_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_attr_destroy_t *) _nanos6_resolve_symbol("alpi_attr_destroy", "alpi", NULL);
	}

	return (*symbol)(attr);
}

int alpi_attr_init(struct alpi_attr *attr)
{
	typedef int alpi_attr_init_t(struct alpi_attr *);

	static alpi_attr_init_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_attr_init_t *) _nanos6_resolve_symbol("alpi_attr_init", "alpi", NULL);
	}

	return (*symbol)(attr);
}

int alpi_attr_size(uint64_t *attr_size)
{
	typedef int alpi_attr_size_t(uint64_t *);

	static alpi_attr_size_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_attr_size_t *) _nanos6_resolve_symbol("alpi_attr_size", "alpi", NULL);
	}

	return (*symbol)(attr_size);
}

int alpi_task_spawn(
	void (*body)(void *),
	void *body_args,
	void (*completion_callback)(void *),
	void *completion_args,
	const char *label,
	const struct alpi_attr *attr)
{
	typedef int alpi_task_spawn_t(
		void (*)(void *), void *,
		void (*)(void *), void *,
		const char *,
		const struct alpi_attr *);

	static alpi_task_spawn_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_task_spawn_t *) _nanos6_resolve_symbol("alpi_task_spawn", "alpi", NULL);
	}

	return (*symbol)(body, body_args, completion_callback, completion_args, label, attr);
}

int alpi_cpu_count(uint64_t *count)
{
	typedef int alpi_cpu_count_t(uint64_t *);

	static alpi_cpu_count_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_cpu_count_t *) _nanos6_resolve_symbol("alpi_cpu_count", "alpi", NULL);
	}

	return (*symbol)(count);
}

int alpi_cpu_logical_id(uint64_t *logical_id)
{
	typedef int alpi_cpu_logical_id_t(uint64_t *);

	static alpi_cpu_logical_id_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_cpu_logical_id_t *) _nanos6_resolve_symbol("alpi_cpu_logical_id", "alpi", NULL);
	}

	return (*symbol)(logical_id);
}

int alpi_cpu_system_id(uint64_t *system_id)
{
	typedef int alpi_cpu_system_id_t(uint64_t *);

	static alpi_cpu_system_id_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (alpi_cpu_system_id_t *) _nanos6_resolve_symbol("alpi_cpu_system_id", "alpi", NULL);
	}

	return (*symbol)(system_id);
}

#pragma GCC visibility pop
