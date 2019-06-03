/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"

#pragma GCC visibility push(default)

int nanos6_in_cluster_mode(void)
{
	typedef int nanos6_in_cluster_mode_t(void);
	
	static nanos6_in_cluster_mode_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_in_cluster_mode_t *) _nanos6_resolve_symbol(
				"nanos6_in_cluster_mode", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_is_master_node(void)
{
	typedef int nanos6_is_master_node_t(void);
	
	static nanos6_is_master_node_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_is_master_node_t *) _nanos6_resolve_symbol(
				"nanos6_is_master_node", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_get_cluster_node_id(void)
{
	typedef int nanos6_get_cluster_node_id_t(void);
	
	static nanos6_get_cluster_node_id_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_cluster_node_id_t *) _nanos6_resolve_symbol(
				"nanos6_get_cluster_node_id", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_get_num_cluster_nodes(void)
{
	typedef int nanos6_get_num_cluster_nodes_t(void);
	
	static nanos6_get_num_cluster_nodes_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_num_cluster_nodes_t *) _nanos6_resolve_symbol(
				"nanos6_get_num_cluster_nodes", "cluster", NULL);
	}
	
	return (*symbol)();
}

void *nanos6_dmalloc(size_t size, nanos6_data_distribution_t policy,
		size_t num_dimensions, size_t *dimensions)
{
	typedef void *nanos6_dmalloc_t(size_t, nanos6_data_distribution_t,
			size_t, size_t *);
	
	static nanos6_dmalloc_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_dmalloc_t *) _nanos6_resolve_symbol(
				"nanos6_dmalloc", "cluster", NULL);
	}
	
	return (*symbol)(size, policy, num_dimensions, dimensions);
}

void nanos6_dfree(void *ptr, size_t size)
{
	typedef void nanos6_dfree_t(void *, size_t);
	
	static nanos6_dfree_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_dfree_t *) _nanos6_resolve_symbol(
				"nanos6_dfree", "cluster", NULL);
	}
	
	(*symbol)(ptr, size);
}

void *nanos6_lmalloc(size_t size)
{
	typedef void *nanos6_lmalloc_t(size_t);
	
	static nanos6_lmalloc_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lmalloc_t *) _nanos6_resolve_symbol(
				"nanos6_lmalloc", "cluster", NULL);
	}
	
	return (*symbol)(size);
}

void nanos6_lfree(void *ptr, size_t size)
{
	typedef void nanos6_lfree_t(void *, size_t);
	
	static nanos6_lfree_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lfree_t *) _nanos6_resolve_symbol(
				"nanos6_lfree", "cluster", NULL);
	}
	
	(*symbol)(ptr, size);
}
