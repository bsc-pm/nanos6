/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "L2Cache.hpp"

std::atomic<uint64_t> L2Cache::_accessedBytes;
std::atomic<uint64_t> L2Cache::_missedBytes;
