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
#include "dependencies/DataAccessType.hpp"

class ClusterNode;
class MemoryPlace;
class MessageTaskNew;
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
	
	//! \brief Propagate satisfiability information to a local task
	//!
	//! \param[in] task the local Task to which we propagate satisfiability
	//! \param[in] satInfo the satisfiability updates we propagate
	void propagateSatisfiability(Task *task, SatisfiabilityInfo const &satInfo);
	
	//! \brief Propagate multiple satisfiability updates to a local task
	//!
	//! \param[in] task the local Task to which we propagate satisfiability
	//! \param[in[ satInfo a vector of SatisfiabilityInfo structs we propagate
	void propagateSatisfiability(Task *task,
			std::vector<SatisfiabilityInfo> const &satInfo);
	
	//! \brief Notify that a region is released on a remote node
	//!
	//! \param[in] offloadedTaskId is the task identifier of the offloader
	//!		node
	//! \param[in] offloader is the cluster node that offloaded the task
	//! \param[in] region is the DataAccessRegion that is released
	//! \param[in] type is the DataAccessType of the associated DataAccess
	//! \param[in] weak is true if the associated DataAccess is weak
	//! \param[in] location is the ClusterMemoryPlace on which the access
	//!		is released
	void sendRemoteAccessRelease(void *offloadedTaskId,
			ClusterNode const *offloader,
			DataAccessRegion const &region,
			DataAccessType type, bool weak,
			MemoryPlace const *location);
	
	//! \brief Release a region of an offloaded Task
	//!
	//! \param[in] task is the offloaded task
	//! \param[in] region is the DataAccessRegion to release
	//! \param[in] type is the DataAcessType of the associated DataAccess
	//! \param[in] weak is true if the associated DataAccess is weak
	//! \param[in] location is the ClusterMemoryPlace on which the access
	//!		is released
	void releaseRemoteAccess(Task *task, DataAccessRegion const &region,
			DataAccessType type, bool weak,
			MemoryPlace const *location);
	
	//! \brief Create and submit a remote task
	//!
	//! \param[in] msg the MessageTaskNew of the remote Task
	void remoteTaskWrapper(MessageTaskNew *msg);
	
	//! \brief Completion callback for a remote task
	//!
	//! \param[in] msg the MessageTaskNew of the remote Task
	void remoteTaskCleanup(MessageTaskNew *msg);
}

#endif // TASK_OFFLOADING_HPP
