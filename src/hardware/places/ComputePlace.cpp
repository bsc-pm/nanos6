/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "ComputePlace.hpp"
#include "MemoryAllocator.hpp"
#include "MemoryPlace.hpp"
#include "tasks/Taskfor.hpp"
#include "hardware-counters/HardwareCounters.hpp"

#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <TaskDataAccessesInfo.hpp>

void ComputePlace::addMemoryPlace(MemoryPlace *mem) {
	_memoryPlaces[mem->getIndex()] = mem;
}

std::vector<int> ComputePlace::getMemoryPlacesIndexes() {
	std::vector<int> indexes(_memoryPlaces.size());

	int i = 0;
	for (memory_places_t::iterator it = _memoryPlaces.begin();
		it != _memoryPlaces.end();
		++it, ++i)
	{
		//indexes.push_back(it->first);
		indexes[i] = it->first;
	}

	return indexes;
}

std::vector<MemoryPlace *> ComputePlace::getMemoryPlaces() {
	std::vector<MemoryPlace *> mems(_memoryPlaces.size());

	int i = 0;
	for (memory_places_t::iterator it = _memoryPlaces.begin();
		it != _memoryPlaces.end();
		++it, ++i)
	{
		//mems.push_back(it->second);
		mems[i] = it->second;
	}

	return mems;
}

ComputePlace::ComputePlace(int index, nanos6_device_t type)
	: _index(index), _type(type), _schedulerData(nullptr)
{
	TaskDataAccessesInfo taskAccessInfo(0);

	// Allocate preallocated taskfor.
	_preallocatedTaskfor = new Taskfor(nullptr, 0, nullptr, nullptr, nullptr, Instrument::task_id_t(), nanos6_task_flag_t::nanos6_final_task, taskAccessInfo, nullptr, true);
	_preallocatedArgsBlockSize = 1024;

	// MemoryAllocator is still not available, so use malloc.
	_preallocatedArgsBlock = malloc(_preallocatedArgsBlockSize);
	FatalErrorHandler::failIf(_preallocatedArgsBlock == nullptr, "Insufficient memory for preallocatedArgsBlock.");

	// Allocate counters in a different call to avoid the free called by
	// getPreallocatedArgsBlock()
	size_t taskCountersSize = HardwareCounters::getTaskHardwareCountersSize();
	TaskHardwareCounters *taskCounters = (TaskHardwareCounters *) malloc(taskCountersSize);
	_preallocatedTaskfor->setHardwareCounters(taskCounters);

	HardwareCounters::taskCreated(_preallocatedTaskfor, true);
}

ComputePlace::~ComputePlace()
{
	Taskfor *taskfor = (Taskfor *) _preallocatedTaskfor;
	assert(taskfor != nullptr);

	TaskHardwareCounters *taskCounters = taskfor->getHardwareCounters();
	assert(taskCounters != nullptr);

	free(taskCounters);

	delete taskfor;

	// First allocation (1024) is done using malloc.
	if (_preallocatedArgsBlockSize == 1024) {
		free(_preallocatedArgsBlock);
	}
	else {
		MemoryAllocator::free(_preallocatedArgsBlock, _preallocatedArgsBlockSize);
	}
}

void *ComputePlace::getPreallocatedArgsBlock(size_t requiredSize)
{
	if (requiredSize > _preallocatedArgsBlockSize) {
		// First allocation (1024) is done using malloc.
		if (_preallocatedArgsBlockSize == 1024) {
			free(_preallocatedArgsBlock);
		}
		else {
			MemoryAllocator::free(_preallocatedArgsBlock, _preallocatedArgsBlockSize);
		}

		_preallocatedArgsBlockSize = requiredSize;
		_preallocatedArgsBlock = MemoryAllocator::alloc(_preallocatedArgsBlockSize);
		FatalErrorHandler::failIf(_preallocatedArgsBlock == nullptr, "Insufficient memory for preallocatedArgsBlock.");
	}
	return _preallocatedArgsBlock;
}
