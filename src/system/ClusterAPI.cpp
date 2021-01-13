/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2021 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/cluster.h>


extern "C" int nanos6_in_cluster_mode(void)
{
	return false;
}

extern "C" int nanos6_is_master_node(void)
{
	return true;
}

extern "C" int nanos6_get_cluster_node_id(void)
{
	return 0;
}

extern "C" int nanos6_get_num_cluster_nodes(void)
{
	return 1;
}

extern "C" void *nanos6_dmalloc(size_t, nanos6_data_distribution_t, size_t, size_t *)
{
	return nullptr;
}

extern "C" void nanos6_dfree(void *, size_t)
{
}

extern "C" void *nanos6_lmalloc(size_t)
{
	return nullptr;
}

extern "C" void nanos6_lfree(void *, size_t)
{
}
