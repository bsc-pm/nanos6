/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentExternalThreadLocalData.hpp"


std::atomic<uint32_t> Instrument::ExternalThreadLocalData::externalThreadCount(0);
