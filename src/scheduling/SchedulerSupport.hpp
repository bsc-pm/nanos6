/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_SUPPORT_HPP
#define SCHEDULER_SUPPORT_HPP

#include <cstdint>

class Task;

namespace SchedulerSupport {

	struct CPUNode {
		uint64_t ticket;
		Task *task;
	};

	constexpr uint64_t roundup(const uint64_t x, const uint64_t y)
	{
		return ((((x) + ((y) - 1ULL)) / (y)) * (y));
	}

	inline uint64_t roundToNextPowOf2(const uint64_t x)
	{
		return roundup(x, 1ULL << (63 - __builtin_clzll(x)));
	}

	inline bool isPowOf2(const uint64_t x)
	{
		return (__builtin_popcountll(x) == 1);
	}
}

#endif // SCHEDULER_SUPPORT_HPP
