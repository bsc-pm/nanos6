/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include "HybridPolicy.hpp"

ConfigVariable<size_t> HybridPolicy::_numBusyIters("cpumanager.busy_iters");
