/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_SCHEDULER_HPP
#define CLUSTER_SCHEDULER_HPP

#include <string>

#include "SchedulerInterface.hpp"
#include "schedulers/cluster/ClusterLocalityScheduler.hpp"
#include "schedulers/cluster/ClusterRandomScheduler.hpp"
#include "system/RuntimeInfo.hpp"

#include <ClusterManager.hpp>

class ClusterScheduler : public SchedulerInterface {
	SchedulerInterface *_clusterSchedulerImplementation;
	
public:
	ClusterScheduler()
	{
		EnvironmentVariable<std::string> clusterSchedulerName("NANOS6_CLUSTER_SCHEDULING_POLICY", "locality");
		
		if (clusterSchedulerName.getValue() == "random") {
			_clusterSchedulerImplementation = new ClusterRandomScheduler();
		} else if (clusterSchedulerName.getValue() == "locality") {
			_clusterSchedulerImplementation = new ClusterLocalityScheduler();
		} else {
			FatalErrorHandler::warnIf(true, "Unknown cluster scheduler:", clusterSchedulerName.getValue(), ". Using default: locality");
			_clusterSchedulerImplementation = new ClusterLocalityScheduler();
		}
	}
	
	~ClusterScheduler()
	{
		delete _clusterSchedulerImplementation;
	}
	
	inline void addReadyTask(Task *task, ComputePlace *computePlace,
			ReadyTaskHint hint = NO_HINT)
	{
		_clusterSchedulerImplementation->addReadyTask(task, computePlace, hint);
	}
	
	inline Task *getReadyTask(ComputePlace *computePlace)
	{
		return _clusterSchedulerImplementation->getReadyTask(computePlace);
	}
	
	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	inline bool hasAvailableWork(ComputePlace *computePlace) const
	{
		return _clusterSchedulerImplementation->hasAvailableWork(computePlace);
	}
	
	inline std::string getName() const
	{
		return _clusterSchedulerImplementation->getName();
	}
};

#endif // CLUSTER_SCHEDULER_HPP
