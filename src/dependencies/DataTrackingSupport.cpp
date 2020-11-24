/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "DataTrackingSupport.hpp"
#include "tasks/Task.hpp"

ConfigVariable<bool> DataTrackingSupport::_NUMASchedulingEnabled("numa.scheduling", true);
const double DataTrackingSupport::_rwBonusFactor = 2.0;
const uint64_t DataTrackingSupport::_distanceThreshold = 15;
const uint64_t DataTrackingSupport::_loadThreshold = 20;
uint64_t DataTrackingSupport::_shouldEnableIS;

bool DataTrackingSupport::shouldEnableIS(Task *task)
{
	return (task->getDataAccesses().getTotalDataSize() <= _shouldEnableIS);
}
