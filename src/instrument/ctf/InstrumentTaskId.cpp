/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "ctfapi/CTFTypes.hpp"
#include "InstrumentTaskId.hpp"

// Reserved ctf2prv task and task type Ids:
//   0 : Idle
//   1 : Runtime
//   2 : Busy Wait
//   3 : Task

std::atomic<ctf_task_id_t> Instrument::task_id_t::_nextTaskId(4);
