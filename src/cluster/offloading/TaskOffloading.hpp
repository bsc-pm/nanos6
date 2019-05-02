/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_OFFLOADING_HPP
#define TASK_OFFLOADING_HPP

#include <cstddef>
#include <sstream>
#include <vector>

#include "SatisfiabilityInfo.hpp"

class ClusterNode;
class Task;

namespace TaskOffloading {
	
	//! \brief Offload a Task to a remote ClusterNode
	//!
	//! \param[in] task is the Task we are offloading
	//! \param[in] satInfo is the sastisfiability info that we already know
	//		about the task already
	//! \param[in] remoteNode is the ClusterNode to which we are offloading
	void offloadTask(Task *task, std::vector<SatisfiabilityInfo> const &satInfo,
			ClusterNode const *remoteNode);
	
	//! \brief Send note that remote task finished
	//!
	//! \param[in] offloadedTaskId is the task identifier on the offloader
	//!		node
	//! \param[in] offloader is the ClusterNode that offloaded the task
	void sendRemoteTaskFinished(void *offloadedTaskId,
			ClusterNode *offloader);
	
	//! \brief Register a remote task
	//!
	//! \param[in] localTask is the remote task (which runs on the current
	//!		node)
	void registerRemoteTask(Task *localTask);
	
	//! \brief Register a remote task and propagate satisfiability info
	//!
	//! \param[in] localTask is the remote task (which runs on the current
	//!		node)
	//! \param[in] satInfo is a std::vector with satisfiability info about the
	//!		remote task
	void registerRemoteTask(Task *localTask,
			std::vector<SatisfiabilityInfo> const &satInfo);
	
	//! \brief Send satisfiability information to an offloaded Task
	//!
	//! \param[in] task is the offloaded task
	//! \param[in] remoteNode is the cluster node, where there is the
	//!		remote task
	//! \param[in] satInfo is the Satisfiability information we are
	//!		sending
	void sendSatisfiability(Task *task, ClusterNode *remoteNode,
			SatisfiabilityInfo const &satInfo);
	
	//! \brief Propagate satisfiability information for a remote task
	//!
	//! \param[in] offloadedTaskId is the task identifier of the offloader
	//!		node
	//! \param[in] offloader is the clusterNode that offloaded the task
	//! \param[in] satInfo is satisfiability info we are propagating
	void propagateSatisfiability(void *offloadedTaskId,
			ClusterNode *offloader, SatisfiabilityInfo const &satInfo);
}

#endif // TASK_OFFLOADING_HPP
