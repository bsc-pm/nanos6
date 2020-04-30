/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"


EnvironmentVariable<bool> HardwareCounters::_verbose("NANOS6_HWCOUNTERS_VERBOSE", false);
EnvironmentVariable<std::string> HardwareCounters::_verboseFile("NANOS6_HWCOUNTERS_VERBOSE_FILE", "nanos6-output-hwcounters.txt");
HardwareCountersInterface *HardwareCounters::_papiBackend;
HardwareCountersInterface *HardwareCounters::_pqosBackend;
std::vector<bool> HardwareCounters::_enabled(HWCounters::NUM_BACKENDS);
std::vector<bool> HardwareCounters::_enabledEvents(HWCounters::TOTAL_NUM_EVENTS);
