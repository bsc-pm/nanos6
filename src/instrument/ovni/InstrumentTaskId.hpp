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
		task_id_t()
		{
		}

		uint32_t assignNewId()
		{
			_taskId = _nextTaskId.fetch_add(1, std::memory_order_relaxed);
			return _taskId;
		}
	};
}

#endif // INSTRUMENT_OVNI_TASK_ID_HPP
