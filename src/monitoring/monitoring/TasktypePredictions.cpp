/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "TasktypePredictions.hpp"

EnvironmentVariable<int> TasktypePredictions::_rollingWindow("NANOS6_MONITORING_ROLLING_WINDOW", 20);
