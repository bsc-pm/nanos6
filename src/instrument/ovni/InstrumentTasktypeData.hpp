/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_TASKTYPE_DATA_HPP
#define INSTRUMENT_OVNI_TASKTYPE_DATA_HPP

#include <atomic>

namespace Instrument {

	struct TasktypeInstrument {
	private:
		static std::atomic<uint32_t> _nextTaskTypeId;
	public:
		uint32_t _taskTypeId;

		TasktypeInstrument() :
			_taskTypeId(0)
		{
		}

		uint32_t assignNewId()
		{
			_taskTypeId = _nextTaskTypeId.fetch_add(1, std::memory_order_relaxed);
			return _taskTypeId;
		}

		bool operator==(TasktypeInstrument const &other) const
		{
			return _taskTypeId == other._taskTypeId;
		}
	};
}


#endif // INSTRUMENT_OVNI_TASKTYPE_DATA_HPP

