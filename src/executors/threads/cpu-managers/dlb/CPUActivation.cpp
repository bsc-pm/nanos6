/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "CPUActivation.hpp"


// 0.1 ms
timespec CPUActivation::_delayCPUEnabling({0, 100000});
