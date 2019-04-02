/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_CLUSTER_H
#define NANOS6_CLUSTER_H

#include "major.h"

#pragma GCC visibility push(default)

// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_cluster_device_api
enum nanos6_cluster_api_t { nanos6_cluster_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Determine whether we are on cluster mode
//!
//! \returns true if we are on cluster mode
int nanos6_in_cluster_mode();

//! \brief Determine whether current node is the master node
//!
//! \returns true if the current node is the master node,
//! otherwise it returns false
int nanos6_is_master_node();

//! \brief Get the id of the current cluster node
//!
//! \returns the id of the current cluster node
int nanos6_get_cluster_node_id();

//! \brief Get the number of cluster nodes
//!
//! \returns the number of cluster nodes
int nanos6_get_num_cluster_nodes();

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* NANOS6_CLUSTER_H */
