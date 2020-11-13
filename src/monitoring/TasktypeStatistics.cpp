/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "TasktypeStatistics.hpp"

ConfigVariable<int> TasktypeStatistics::_rollingWindow("monitoring.rolling_window", 20);
