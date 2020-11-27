/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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
		ConfigVariable<std::string> clusterSchedulerName("cluster.scheduling_policy");

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

	inline void addReadyTasks(
		nanos6_device_t taskType,
		Task *tasks[],
		const size_t numTasks,
		ComputePlace *computePlace,
		ReadyTaskHint hint)
	{
		_clusterSchedulerImplementation->addReadyTasks(taskType, tasks, numTasks, computePlace, hint);
	}

	inline Task *getReadyTask(ComputePlace *computePlace)
	{
		return _clusterSchedulerImplementation->getReadyTask(computePlace);
	}

	inline bool isServingTasks() const
	{
		return _clusterSchedulerImplementation->isServingTasks();
	}

	inline std::string getName() const
	{
		return _clusterSchedulerImplementation->getName();
	}
};

#endif // CLUSTER_SCHEDULER_HPP
