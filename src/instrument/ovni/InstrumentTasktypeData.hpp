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
		uint32_t id;

		TasktypeInstrument()
		{
			id = 0;
		}

		uint32_t autoAssingId()
		{
			id = _nextTaskTypeId++;
			return id;
		}

		bool operator==(TasktypeInstrument const &other) const
		{
			return id == other.id;
		}
	};
}


#endif // INSTRUMENT_OVNI_TASKTYPE_DATA_HPP

