/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"

#pragma GCC visibility push(default)

int nanos6_in_cluster_mode()
{
	typedef int nanos6_in_cluster_mode_t();
	
	static nanos6_in_cluster_mode_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_in_cluster_mode_t *) _nanos6_resolve_symbol(
				"nanos6_in_cluster_mode", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_is_master_node()
{
	typedef int nanos6_is_master_node_t();
	
	static nanos6_is_master_node_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_is_master_node_t *) _nanos6_resolve_symbol(
				"nanos6_is_master_node", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_get_cluster_node_id()
{
	typedef int nanos6_get_cluster_node_id_t();
	
	static nanos6_get_cluster_node_id_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_cluster_node_id_t *) _nanos6_resolve_symbol(
				"nanos6_get_cluster_node_id", "cluster", NULL);
	}
	
	return (*symbol)();
}

int nanos6_get_num_cluster_nodes()
{
	typedef int nanos6_get_num_cluster_nodes_t();
	
	static nanos6_get_num_cluster_nodes_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_num_cluster_nodes_t *) _nanos6_resolve_symbol(
				"nanos6_get_num_cluster_nodes", "cluster", NULL);
	}
	
	return (*symbol)();
}
