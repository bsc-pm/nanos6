/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASKTYPE_DATA_HPP
#define INSTRUMENT_CTF_TASKTYPE_DATA_HPP

#include <atomic>

#include "ctfapi/CTFTypes.hpp"

namespace Instrument {

	struct TasktypeInstrument {
	private:
		static std::atomic<ctf_tasktype_id_t> _nextTaskTypeId;
	public:
		ctf_tasktype_id_t id;

		TasktypeInstrument()
		{
			id = 0;
		}

		ctf_tasktype_id_t autoAssingId()
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


#endif // INSTRUMENT_CTF_TASKTYPE_DATA_HPP

