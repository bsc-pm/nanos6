/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <utility>
#include <map>
#include <nanos6/task-instantiation.h>
#include <vector>

#include "ClusterTaskContext.hpp"
#include "TaskOffloading.hpp"
#include "src/executors/threads/WorkerThread.hpp"
#include "src/lowlevel/PaddedSpinLock.hpp"
#include "src/tasks/Task.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegistration.hpp>
#include <Directory.hpp>
#include <MessageReleaseAccess.hpp>
#include <MessageSatisfiability.hpp>
#include <MessageTaskFinished.hpp>
#include <MessageTaskNew.hpp>

namespace TaskOffloading {
	
	//! Information for tasks that have been offloaded to the
	//! current node. Objects of this type are used to temporarily
	//! keep satisfiability info the might arrive ahead of the
	//! creation of the actual task.
	struct RemoteTaskInfo {
		Task *_localTask;
		std::vector<SatisfiabilityInfo> _satInfo;
		PaddedSpinLock<64> _lock;
		
		RemoteTaskInfo()
			: _localTask(nullptr),
			_satInfo(),
			_lock()
		{
		}
	};
	
	//! When a ClusterNode offloads a task, it attaches an id that is unique
	//! on the offloading node, so we can create a mapping between an
	//! offloaded Task and the matching remote Task.
	//!
	//! Here, we use this id as an index to a container to retrieve
	//! the local information of the remote task.
	struct RemoteTasks {
		typedef std::pair<void *, int> remote_index_t;
		typedef std::map<remote_index_t, RemoteTaskInfo> remote_map_t;
		
		//! The actual map holding the remote tasks' info
		remote_map_t _taskMap;
		
		//! Lock to protect access to the map
		PaddedSpinLock<64> _lock;
		
		RemoteTasks()
			: _taskMap(),
			_lock()
		{
		}
		
		//! This will return a reference to the RemoteTaskInfo entry
		//! within this map. If this is the first access to this entry
		//! we will create it and return a reference to the new
		//! RemoteTaskInfo object
		RemoteTaskInfo &getTaskInfo(void *offloadedTaskId,
				int offloaderId)
		{
			auto key = std::make_pair(offloadedTaskId, offloaderId);
			
			std::lock_guard<PaddedSpinLock<64>> guard(_lock);
			return _taskMap[key];
		}
		
		//! This erases a map entry. It assumes that there is already
		//! an entry with the given key
		void eraseTaskInfo(void *offloadTaskId, int offloaderId)
		{
			auto key = std::make_pair(offloadTaskId, offloaderId);
			
			std::lock_guard<PaddedSpinLock<64>> guard(_lock);
			
			assert(_taskMap.find(key) != _taskMap.end());
			_taskMap.erase(key);
		}
	};
	
	//! This is our map for all the remote tasks, currently on the node
	static RemoteTasks _remoteTasks;
	
	void propagateSatisfiability(Task *localTask,
			SatisfiabilityInfo const &satInfo)
	{
		assert(localTask != nullptr);
		assert(!satInfo.empty());
		
		WorkerThread *currentThread =
			WorkerThread::getCurrentWorkerThread();
		
		CPU *cpu = (currentThread == nullptr) ? nullptr :
				currentThread->getComputePlace();
		
		CPUDependencyData localDependencyData;
		CPUDependencyData &hpDependencyData = (cpu != nullptr) ?
			cpu->getDependencyData() : localDependencyData;
		
		MemoryPlace const *loc;
		if (!Directory::isDirectoryMemoryPlace(satInfo._src)) {
			loc = ClusterManager::getMemoryNode(satInfo._src);
			DataAccessRegistration::propagateSatisfiability(
					localTask, satInfo._region, cpu,
					hpDependencyData, satInfo._readSat,
					satInfo._writeSat, loc);
			
			return;
		}
		
		//! The access is in the Directory. Retrieve the home nodes and
		//! propagate satisfiability per region
		Directory::HomeNodesArray *array =
			Directory::find(satInfo._region);
		assert(!array->empty());
		
		for (HomeMapEntry const *entry : *array) {
			loc = entry->getHomeNode();
			DataAccessRegion entryRegion =
				entry->getAccessRegion();
			DataAccessRegion subRegion =
				satInfo._region.intersect(entryRegion);
			
			DataAccessRegistration::propagateSatisfiability(
				localTask, subRegion, cpu, hpDependencyData,
				satInfo._readSat, satInfo._writeSat, loc);
		}
		
		delete array;
	}
	
	void propagateSatisfiability(Task *localTask,
			std::vector<SatisfiabilityInfo> const &satInfo)
	{
		assert(localTask != nullptr);
		assert(!satInfo.empty());
		
		for (SatisfiabilityInfo const &sat : satInfo) {
			propagateSatisfiability(localTask, sat);
		}
	}
	
	static void unregisterRemoteTask(void *offloadedTaskId,
			ClusterNode *offloader)
	{
		assert(offloader != nullptr);
		_remoteTasks.eraseTaskInfo(offloadedTaskId,
				offloader->getIndex());
	}
	
	void offloadTask(Task *task, std::vector<SatisfiabilityInfo> const &satInfo,
			ClusterNode const *remoteNode)
	{
		assert(task != nullptr);
		assert(remoteNode != nullptr);
		
		ClusterNode const *thisNode = ClusterManager::getCurrentClusterNode();
		nanos6_task_info_t *taskInfo = task->getTaskInfo();
		nanos6_task_invocation_info_t *taskInvocationInfo =
			task->getTaskInvokationInfo();
		size_t flags = task->getFlags();
		void *argsBlock = task->getArgsBlock();
		size_t argsBlockSize = task->getArgsBlockSize();
		size_t nrSatInfo = satInfo.size();
		SatisfiabilityInfo const *satInfoPtr = (nrSatInfo == 0) ? nullptr :
						satInfo.data();
		
		MessageTaskNew *msg =
			new MessageTaskNew(thisNode, taskInfo,
					taskInvocationInfo, flags,
					taskInfo->implementation_count,
					taskInfo->implementations, nrSatInfo,
					satInfoPtr, argsBlockSize, argsBlock,
					(void *)task);
		
		ClusterManager::sendMessage(msg, remoteNode);
	}
	
	void sendRemoteTaskFinished(void *offloadedTaskId,
			ClusterNode *offloader)
	{
		unregisterRemoteTask(offloadedTaskId, offloader);
		MessageTaskFinished *msg =
			new MessageTaskFinished(
				ClusterManager::getCurrentClusterNode(),
				offloadedTaskId);
		
		ClusterManager::sendMessage(msg, offloader);
	}
	
	void sendSatisfiability(Task *task, ClusterNode *remoteNode,
			SatisfiabilityInfo const &satInfo)
	{
		assert(task != nullptr);
		assert(remoteNode != nullptr);
		assert(!satInfo.empty());
		
		ClusterNode *current = ClusterManager::getCurrentClusterNode();
		MessageSatisfiability *msg =
			new MessageSatisfiability(current, (void *)task,
					satInfo);
		
		ClusterManager::sendMessage(msg, remoteNode);
	}
	
	void propagateSatisfiability(void *offloadedTaskId, ClusterNode *offloader,
			SatisfiabilityInfo const &satInfo)
	{
		RemoteTaskInfo &taskInfo =
			_remoteTasks.getTaskInfo(offloadedTaskId,
					offloader->getIndex());
		
		taskInfo._lock.lock();
		if (taskInfo._localTask == nullptr) {
			//! The remote task has not been created yet, so we
			//! just add the info to the temporary vector
			taskInfo._satInfo.push_back(satInfo);
			taskInfo._lock.unlock();
		} else {
			//! We *HAVE* to leave the lock now, because propagating
			//! satisfiability might lead to unregistering the remote
			//! task
			taskInfo._lock.unlock();
			propagateSatisfiability(taskInfo._localTask, satInfo);
		}
	}
	
	void sendRemoteAccessRelease(void *offloadedTaskId,
			ClusterNode const *offloader,
			DataAccessRegion const &region, DataAccessType type,
			bool weak, MemoryPlace const *location)
	{
		assert(location != nullptr);
		
		/* If location is a host device on this node it is a cluster
		 * device from the point of view of the remote node */
		if (location->getType() == nanos6_host_device) {
			location = ClusterManager::getCurrentMemoryNode();
		}
		
		ClusterNode *current = ClusterManager::getCurrentClusterNode();
		MessageReleaseAccess *msg =
			new MessageReleaseAccess(current, offloadedTaskId,
					region, type, weak,
					location->getIndex());
		
		ClusterManager::sendMessage(msg, offloader);
	}
	
	void releaseRemoteAccess(Task *task, DataAccessRegion const &region,
			DataAccessType type, bool weak, MemoryPlace const *location)
	{
		assert(task != nullptr);
		assert(location->getType() == nanos6_cluster_device);
		
		WorkerThread *currentThread =
			WorkerThread::getCurrentWorkerThread();
		
		CPU *cpu = (currentThread == nullptr) ? nullptr :
				currentThread->getComputePlace();
		
		CPUDependencyData localDependencyData;
		CPUDependencyData &hpDependencyData = (cpu != nullptr) ?
			cpu->getDependencyData() : localDependencyData;
		
		DataAccessRegistration::releaseAccessRegion(task, region, type,
				weak, cpu, hpDependencyData, location);
	}
	
	void remoteTaskWrapper(MessageTaskNew *msg)
	{
		assert(msg != nullptr);
		
		ClusterNode *offloader =
			ClusterManager::getClusterNode(msg->getSenderId());
		
		nanos6_task_info_t *taskInfo = msg->getTaskInfo();
		void *offloadedTaskId = msg->getOffloadedTaskId();
		
		size_t numTaskImplementations;
		nanos6_task_implementation_info_t *taskImplementations =
			msg->getImplementations(numTaskImplementations);
		
		taskInfo->implementations = taskImplementations;
		nanos6_task_invocation_info_t *taskInvocationInfo =
			msg->getTaskInvocationInfo();
		
		size_t argsBlockSize;
		void *argsBlock = msg->getArgsBlock(argsBlockSize);
		
		size_t flags = msg->getFlags();
		
		Task *task;
		void *newArgsBlock;
		nanos6_create_task(taskInfo, taskInvocationInfo, argsBlockSize,
				&newArgsBlock, (void **)&task, flags, 0);
		
		if (argsBlockSize != 0) {
			memcpy(newArgsBlock, argsBlock, argsBlockSize);
		}
		
		task->markAsRemote();
		TaskOffloading::ClusterTaskContext *clusterContext =
			new TaskOffloading::ClusterTaskContext(
					offloadedTaskId, offloader);
		
		task->setClusterContext(clusterContext);
		
		//! Register remote Task with TaskOffloading mechanism before
		//! submitting it to the dependency system.
		RemoteTaskInfo &remoteTaskInfo =
			_remoteTasks.getTaskInfo(offloadedTaskId,
					offloader->getIndex());
		
		std::lock_guard<PaddedSpinLock<64>> lock(remoteTaskInfo._lock);
		assert(remoteTaskInfo._localTask == nullptr);
		remoteTaskInfo._localTask = task;
		
		nanos6_submit_task(task);
		
		//! propagate satisfiability embedded in the Message
		size_t numSatInfo;
		TaskOffloading::SatisfiabilityInfo *satInfo =
			msg->getSatisfiabilityInfo(numSatInfo);
		for (size_t i = 0; i < numSatInfo; ++i) {
			propagateSatisfiability(task, satInfo[i]);
		}
		
		//! propagate, also any satisfiability that has already arrived
		if (!remoteTaskInfo._satInfo.empty()) {
			propagateSatisfiability(task, remoteTaskInfo._satInfo);
			remoteTaskInfo._satInfo.clear();
		}
	}
	
	void remoteTaskCleanup(MessageTaskNew *msg)
	{
		assert(msg != nullptr);
		
		void *offloadedTaskId = msg->getOffloadedTaskId();
		ClusterNode *offloader =
			ClusterManager::getClusterNode(msg->getSenderId());
		
		sendRemoteTaskFinished(offloadedTaskId, offloader);
		
		//! For the moment, we do not delete the Message since it includes the
		//! buffers that hold the nanos6_task_info_t and the
		//! nanos6_task_implementation_info_t which we might need later on,
		//! e.g. Extrae is using these during shutdown. This will change once
		//! mercurium gives us access to the respective fields within the
		//! binary.
		//! delete msg;
	}
}
