/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "TasktypePredictions.hpp"

ConfigVariable<int> TasktypePredictions::_rollingWindow("monitoring.rolling_window", 20);
