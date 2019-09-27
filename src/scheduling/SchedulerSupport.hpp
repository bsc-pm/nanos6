/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_SUPPORT_HPP
#define SCHEDULER_SUPPORT_HPP

#include <cstdint>

#include "scheduling/ReadyQueue.hpp"

class ComputePlace;
class Task;

struct TaskSchedulingInfo {
	Task *_task;
	ComputePlace *_computePlace;
	ReadyTaskHint _hint;
	
	TaskSchedulingInfo(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
		:  _task(task), _computePlace(computePlace), _hint(hint)
	{
	}
};

struct CPUNode {
	uint64_t ticket;
	Task *task;
};


constexpr static uint64_t roundup(uint64_t const x, uint64_t const y) {
	return ((((x) + ((y) - 1ULL)) / (y)) * (y));
}

static inline uint64_t roundToNextPowOf2(uint64_t const x) {
	return roundup(x, 1ULL << (63 - __builtin_clzll(x)));
}

static inline bool isPowOf2(uint64_t const x) {
	return (__builtin_popcountll(x) == 1);
}

#endif // SCHEDULER_SUPPORT_HPP
