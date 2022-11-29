/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#include "DeviceUnsyncScheduler.hpp"

Task *DeviceUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	return regularGetReadyTask(computePlace);
}
