/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_ID_HPP
#define INSTRUMENT_CTF_TASK_ID_HPP

#include <nanos6.h>
#include <atomic>
#include <cstdint>

#include "ctfapi/CTFTypes.hpp"

namespace Instrument {

	struct task_id_t {
	private:
		static std::atomic<ctf_task_id_t> _nextTaskId;
	public:
		ctf_task_id_t _taskId; // TODO use the ctf typedef for task id
		task_id_t() {}

		// task_id_t are created in other parts of the runtime apart
		// from when adding tasks. We want the numeric task ID to be
		// assigned only to tasks, and to do so we use the dummy bool
		// arguemnt to distinguish when a numeric task id must be
		// generated.
		task_id_t(bool)
		{
			_taskId = _nextTaskId++;
		}
	};
}

#endif // INSTRUMENT_CTF_TASK_ID_HPP
