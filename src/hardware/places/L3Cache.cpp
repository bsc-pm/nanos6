/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "L3Cache.hpp"

double L3Cache::_penalty;
std::atomic<uint64_t> L3Cache::_accessedBytes;
std::atomic<uint64_t> L3Cache::_missedBytes;
