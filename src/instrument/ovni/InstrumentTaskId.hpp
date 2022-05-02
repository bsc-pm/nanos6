/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_TASK_ID_HPP
#define INSTRUMENT_OVNI_TASK_ID_HPP

#include <atomic>
#include <cstdint>

namespace Instrument {

	struct task_id_t {
	private:
		static std::atomic<uint32_t> _nextTaskId;
	public:
		uint32_t _taskId;
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

#endif // INSTRUMENT_OVNI_TASK_ID_HPP
