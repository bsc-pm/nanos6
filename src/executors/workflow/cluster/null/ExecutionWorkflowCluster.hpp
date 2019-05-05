#ifndef EXECUTION_WORKFLOW_CLUSTER_HPP
#define EXECUTION_WORKFLOW_CLUSTER_HPP

#include <functional>

#include "../../ExecutionStep.hpp"

class ComputePlace;
class MemoryPlace;
class DataAccess;

namespace ExecutionWorkflow {
	class ClusterAllocationAndPinningStep : public Step {
	public:
		ClusterAllocationAndPinningStep(
			__attribute__((unused)) RegionTranslation &regionTranslation,
			__attribute__((unused)) MemoryPlace const *memoryPlace
		) : Step()
		{
		}
	};
	
	class ClusterDataLinkStep : public DataLinkStep {
		ClusterDataLinkStep(
			__attribute__((unused)) MemoryPlace const *sourceMemoryPlace,
			__attribute__((unused)) MemoryPlace const *targetMemoryPlace,
			__attribute__((unused)) DataAccess *access
		) : DataLinkStep(access)
		{
		}
	};
	
	class ClusterDataCopyStep : public Step {
	public:
		ClusterDataCopyStep(
			__attribute__((unused)) MemoryPlace const *sourceMemoryPlace,
			__attribute__((unused)) MemoryPlace const *targetMemoryPlace,
			__attribute__((unused)) DataAccess const *access,
			__attribute__((unused)) RegionTranslation const &targetTranslation
		) : Step()
		{
		}
	};
	
	class ClusterExecutionStep : public Step {
	public:
		ClusterExecutionStep(
			__attribute__((unused)) Task *task,
			__attribute__((unused)) ComputePlace *computePlace
		) : Step()
		{
		}
	};
	
	class ClusterNotificationStep : public Step {
	public:
		ClusterNotificationStep(
			__attribute__((unused)) std::function<void ()> const &callback
		) : Step()
		{
		}
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
	
	class ClusterDataReleaseStep : public DataReleaseStep {
	public:
		ClusterDataReleaseStep(
			__attribute__((unused)) TaskOffloading::ClusterTaskContext *context,
			__attribute__((unused)) DataAccess *access
		) : DataReleaseStep(access)
		{
		}
	};
	
	inline Step *clusterCopy(
		__attribute__((unused)) MemoryPlace const *source,
		__attribute__((unused)) MemoryPlace const *target,
		__attribute__((unused)) RegionTranslation const &translation,
		__attribute__((unused)) DataAccess *access
	) {
		return new Step();
	}
}


#endif /* EXECUTION_WORKFLOW_Cluster_HPP */
