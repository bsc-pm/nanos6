/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "ComputePlace.hpp"
#include "MemoryAllocator.hpp"
#include "MemoryPlace.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "tasks/Taskfor.hpp"

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

ComputePlace::ComputePlace(int index, nanos6_device_t type, bool owned) :
	_owned(owned),
	_index(index),
	_type(type)
{
	TaskDataAccessesInfo taskAccessInfo(0);

	void *taskCountersAddress = nullptr;
	size_t taskCountersSize = TaskHardwareCounters::getAllocationSize();

	// Allocate hardware counters in a different call to avoid
	// the free by getPreallocatedArgsBlock()
	if (taskCountersSize > 0) {
		taskCountersAddress = malloc(taskCountersSize);
		assert(taskCountersAddress != nullptr);
	}

	// Allocate task monitoring statistics
	size_t taskStatisticsSize = Monitoring::getTaskStatisticsSize();
	TaskStatistics *taskStatistics = (TaskStatistics *) malloc(taskStatisticsSize);
	_preallocatedTaskfor->setTaskStatistics(taskStatistics);

	// Allocate preallocated taskfor
	_preallocatedTaskfor = new Taskfor(nullptr, 0, nullptr, nullptr, nullptr,
		Instrument::task_id_t(), nanos6_task_flag_t::nanos6_final_task,
		taskAccessInfo, taskCountersAddress, nullptr, true);
	_preallocatedArgsBlockSize = 1024;

	// MemoryAllocator is still not available, so use malloc
	_preallocatedArgsBlock = malloc(_preallocatedArgsBlockSize);
	FatalErrorHandler::failIf(_preallocatedArgsBlock == nullptr,
		"Insufficient memory for preallocatedArgsBlock");

	HardwareCounters::taskCreated(_preallocatedTaskfor);
}

ComputePlace::~ComputePlace()
{
	Taskfor *taskfor = (Taskfor *) _preallocatedTaskfor;
	assert(taskfor != nullptr);

	// Retreive the allocation address
	const TaskHardwareCounters &taskCounters = taskfor->getHardwareCounters();
	void *allocationAddress = taskCounters.getAllocationAddress();

	// Free task statistics
	TaskStatistics *taskStatistics = taskfor->getTaskStatistics();
	assert(taskStatistics != nullptr);

	free(taskStatistics);

	delete taskfor;

	// After hardware counters are deleted (task destructor), free the
	// task hardware counters structure if existent
	if (allocationAddress != nullptr) {
		free(allocationAddress);
	}

	// First allocation (1024) is done using malloc
	if (_preallocatedArgsBlockSize == 1024) {
		free(_preallocatedArgsBlock);
	} else {
		MemoryAllocator::free(_preallocatedArgsBlock, _preallocatedArgsBlockSize);
	}
}

void *ComputePlace::getPreallocatedArgsBlock(size_t requiredSize)
{
	if (requiredSize > _preallocatedArgsBlockSize) {
		// First allocation (1024) was done using malloc
		if (_preallocatedArgsBlockSize == 1024) {
			free(_preallocatedArgsBlock);
		} else {
			MemoryAllocator::free(_preallocatedArgsBlock, _preallocatedArgsBlockSize);
		}

		_preallocatedArgsBlockSize = requiredSize;
		_preallocatedArgsBlock = MemoryAllocator::alloc(_preallocatedArgsBlockSize);
		FatalErrorHandler::failIf(_preallocatedArgsBlock == nullptr,
			"Insufficient memory for preallocatedArgsBlock");
	}
	return _preallocatedArgsBlock;
}
