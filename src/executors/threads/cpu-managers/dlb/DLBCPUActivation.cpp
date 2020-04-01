/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "DLBCPUActivation.hpp"


// 0.1 ms
timespec DLBCPUActivation::_delayCPUEnabling({0, 100000});
std::atomic<size_t> DLBCPUActivation::_numActiveOwnedCPUs(0);
