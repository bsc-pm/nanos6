#ifndef EXECUTION_WORKFLOW_CLUSTER_HPP
#define EXECUTION_WORKFLOW_CLUSTER_HPP

#include <functional>

#include "../ExecutionStep.hpp"

#include <ClusterManager.hpp>
#include <ClusterTaskContext.hpp>
#include <DataAccess.hpp>
#include <Directory.hpp>
#include <InstrumentLogMessage.hpp>
#include <SatisfiabilityInfo.hpp>
#include <TaskOffloading.hpp>
#include <VirtualMemoryManagement.hpp>

class ComputePlace;
class MemoryPlace;

namespace ExecutionWorkflow {
	class ClusterAllocationAndPinningStep : public Step {
	public:
		ClusterAllocationAndPinningStep(
			__attribute__((unused)) RegionTranslation &regionTranslation,
			__attribute__((unused)) MemoryPlace const *memoryPlace
		) : Step ()
		{
		}
	};
	
	class ClusterDataLinkStep : public DataLinkStep {
		//! The MemoryPlace that holds the data at the moment
		MemoryPlace const *_sourceMemoryPlace;
		
		//! The MemoryPlace that requires the data
		MemoryPlace const *_targetMemoryPlace;
		
		//! DataAccessRegion that the Step covers
		DataAccessRegion _region;
		
		//! The task in which the access belongs to
		Task *_task;
		
		//! read satisfiability at creation time
		bool _read;
		
		//! write satisfiability at creation time
		bool _write;
		
	public:
		ClusterDataLinkStep(
			MemoryPlace const *sourceMemoryPlace,
			MemoryPlace const *targetMemoryPlace,
			DataAccess *access
		) : DataLinkStep(access),
			_sourceMemoryPlace(sourceMemoryPlace),
			_targetMemoryPlace(targetMemoryPlace),
			_region(access->getAccessRegion()),
			_task(access->getOriginator()),
			_read(access->readSatisfied()),
			_write(access->writeSatisfied())
		{
			access->setDataLinkStep(this);
		}
		
		void linkRegion(
			DataAccessRegion const &region,
			MemoryPlace const *location,
			bool read,
			bool write
		);
		
		//! Start the execution of the Step
		void start();
	};
	
	class ClusterDataCopyStep : public Step {
		//! The MemoryPlace that the data will be copied from.
		MemoryPlace const *_sourceMemoryPlace;
		
		//! The MemoryPlace that the data will be copied to.
		MemoryPlace const *_targetMemoryPlace;
		
		//! A mapping of the address range in the source node to the target node.
		RegionTranslation _targetTranslation;
		
	public:
		ClusterDataCopyStep(
			MemoryPlace const *sourceMemoryPlace,
			MemoryPlace const *targetMemoryPlace,
			RegionTranslation const &targetTranslation
		) : Step(),
			_sourceMemoryPlace(sourceMemoryPlace),
			_targetMemoryPlace(targetMemoryPlace),
			_targetTranslation(targetTranslation)
		{
		}
		
		//! Start the execution of the Step
		void start();
	};
	
	class ClusterDataReleaseStep : public DataReleaseStep {
		//! identifier of the remote task
		void *_remoteTaskIdentifier;
		
		//! the cluster node we need to notify
		ClusterNode const *_offloader;
		
	public:
		ClusterDataReleaseStep(
			TaskOffloading::ClusterTaskContext *context,
			DataAccess *access
		) : DataReleaseStep(access),
			_remoteTaskIdentifier(context->getRemoteIdentifier()),
			_offloader(context->getRemoteNode())
		{
			access->setDataReleaseStep(this);
		}
		
		void releaseRegion(DataAccessRegion const &region,
			MemoryPlace const *location);
		
		bool checkDataRelease(DataAccess const *access);
		
		void start();
	};
	
	class ClusterExecutionStep : public Step {
		std::vector<TaskOffloading::SatisfiabilityInfo> _satInfo;
		ClusterNode *_remoteNode;
		Task *_task;
		
	public:
		ClusterExecutionStep(Task *task, ComputePlace *computePlace);
		
		//! Inform the execution Step about the existence of a
		//! pending data copy.
		//!
		//! \param[in] source is the id of the MemoryPlace that the data
		//!            is currently located
		//! \param[in] region is the memory region being copied
		//! \param[in] size is the size of the region being copied.
		//! \param[in] read is true if access is read-satisfied
		//! \param[in] write is true if access is write-satisfied
		void addDataLink(int source, DataAccessRegion const &region,
			bool read, bool write);
		
		//! Start the execution of the Step
		void start();
	};
	
	class ClusterNotificationStep : public Step {
		std::function<void ()> const _callback;
		
	public:
		ClusterNotificationStep(std::function<void ()> const &callback) :
			Step(), _callback(callback)
		{
		}
		
		//! Start the execution of the Step
		void start();
	};
	
	class ClusterUnpinningStep : public Step {
	public:
		ClusterUnpinningStep(
			__attribute__((unused)) MemoryPlace const *targetMemoryPlace,
			__attribute__((unused)) RegionTranslation const &targetTranslation
		) : Step()
		{
		}
	};
	
	inline Step *clusterFetchData(
		MemoryPlace const *source,
		MemoryPlace const *target,
		RegionTranslation const &translation,
		DataAccess *access
	) {
		assert(source != nullptr);
		nanos6_device_t sourceType = source->getType();
		assert(target == ClusterManager::getCurrentMemoryNode());
		
		//! Currently, we cannot have a cluster data copy where the source
		//! location is in the Directory. This would mean that the data
		//! have not been written yet (that's why they're not in a
		//! non-directory location), so we are reading something that is
		//! not initialized yet
		assert(!Directory::isDirectoryMemoryPlace(source) &&
			"You're probably trying to read something "
			"that has not been initialized yet!");
		
		//! The source device is a host MemoryPlace of the current
		//! ClusterNode. We do not really need to perform a
		//! DataTransfer
		if ((sourceType == nanos6_host_device)) {
			return new Step();
		}
		
		assert(source->getType() == nanos6_cluster_device);
		DataAccessObjectType objectType = access->getObjectType();
		DataAccessType type = access->getType();
		DataAccessRegion region = access->getAccessRegion();
		bool isDistributedRegion =
			VirtualMemoryManagement::isDistributedRegion(region);
		
		bool needsTransfer =
			(
			 	//! We need a DataTransfer for a taskwait access
				//! in the following cases:
				//! 1) the access is not a NO_ACCESS_TYPE, so it
				//!    is part of the calling task's dependencies,
				//!    which means that the latest version of
				//!    the region needs to be present in the
				//!    context of the task at all times.
				//! 2) the access is a NO_ACCESS_TYPE access, so
				//!    it represents a region allocated within
				//!    the context of the Task but it is local
				//!    memory, so it needs to be present in the
				//!    context of the Task after the taskwait.
				//!    Distributed memory regions, do not need
				//!    to trigger a DataCopy, since anyway can
				//!    only be accessed from within subtasks.
				//!
				//! In both cases, we can avoid the copy if the
				//! access is a read-only access.
			 	(objectType == taskwait_type)
				&& (type != READ_ACCESS_TYPE)
				&& ((type != NO_ACCESS_TYPE) || !isDistributedRegion)
			) ||
			(
				//! We need a DataTransfer for an access_type
				//! access, if the access is not read-only
			 	(objectType == access_type)
				&& (type != WRITE_ACCESS_TYPE)
			);
		
		if (needsTransfer) {
			return new ClusterDataCopyStep(source, target, translation);
		}
		
		return new Step();
	}
	
	inline Step *clusterLinkData(
		MemoryPlace const *source,
		MemoryPlace const *target,
		__attribute__((unused)) RegionTranslation const &translation,
		DataAccess *access
	) {
		assert(access->getObjectType() == access_type);
		return new ClusterDataLinkStep(source, target, access);
	}
	
	inline Step *clusterCopy(
		MemoryPlace const *source,
		MemoryPlace const *target,
		RegionTranslation const &translation,
		DataAccess *access
	) {
		assert(target != nullptr);
		assert(access != nullptr);
		
		ClusterMemoryNode *current =
			ClusterManager::getCurrentMemoryNode();
		if (target->getType() != nanos6_cluster_device) {
			//! At the moment cluster copies take into account only
			//! Cluster and host devices
			assert(target->getType() == nanos6_host_device);
			assert(!Directory::isDirectoryMemoryPlace(target));
			target = current;
		}
		
		if (target == current) {
			return clusterFetchData(source, target, translation, access);
		} else {
			return clusterLinkData(source, target, translation, access);
		}
	}
}


#endif // EXECUTION_WORKFLOW_CLUSTER_HPP
