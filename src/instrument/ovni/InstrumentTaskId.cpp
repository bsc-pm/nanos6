/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentTaskId.hpp"

std::atomic<uint32_t> Instrument::task_id_t::_nextTaskId(1);
