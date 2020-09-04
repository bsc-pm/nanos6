/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentCPULocalData.hpp"
#include <instrument/support/InstrumentCPULocalDataSupport.hpp>

static Instrument::CPULocalData *_virtualCPULocalData = nullptr;

Instrument::CPULocalData *Instrument::getCTFCPULocalData()
{
	assert(_virtualCPULocalData != nullptr);
	CPULocalData *cpuLocalData = getCPULocalData();
	return (cpuLocalData == nullptr) ? _virtualCPULocalData : cpuLocalData;
}

Instrument::CPULocalData *Instrument::getCTFVirtualCPULocalData()
{
	return _virtualCPULocalData;
}

void Instrument::setCTFVirtualCPULocalData(Instrument::CPULocalData *virtualCPULocalData)
{
	assert(virtualCPULocalData != nullptr);
	assert(_virtualCPULocalData == nullptr);
	_virtualCPULocalData = virtualCPULocalData;
}
