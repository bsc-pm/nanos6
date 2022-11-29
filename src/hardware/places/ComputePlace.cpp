/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#include "ComputePlace.hpp"
#include "MemoryAllocator.hpp"
#include "MemoryPlace.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "support/BitManipulation.hpp"

#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <TaskDataAccessesInfo.hpp>

void ComputePlace::addMemoryPlace(MemoryPlace *mem) {
	_memoryPlaces[mem->getIndex()] = mem;
}

std::vector<int> ComputePlace::getMemoryPlacesIndexes()
{
	std::vector<int> indexes(_memoryPlaces.size());

	int i = 0;
	memory_places_t::iterator it;
	for (it = _memoryPlaces.begin(); it != _memoryPlaces.end(); ++it, ++i) {
		indexes[i] = it->first;
	}

	return indexes;
}

std::vector<MemoryPlace *> ComputePlace::getMemoryPlaces()
{
	std::vector<MemoryPlace *> mems(_memoryPlaces.size());

	int i = 0;
	memory_places_t::iterator it;
	for (it = _memoryPlaces.begin(); it != _memoryPlaces.end(); ++it, ++i) {
		mems[i] = it->second;
	}

	return mems;
}
