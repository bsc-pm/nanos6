/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "DataTrackingSupport.hpp"
#include "hardware/HardwareInfo.hpp"

uint64_t DataTrackingSupport::IS_THRESHOLD;

size_t DataTrackingSupport::getNUMATrackingNodes()
{
    if (_NUMATrackingEnabled) {
        return HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
    } else {
        return 1;
    }
}
