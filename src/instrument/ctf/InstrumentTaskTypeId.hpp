/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_TYPE_ID_HPP
#define INSTRUMENT_CTF_TASK_TYPE_ID_HPP

#include <atomic>

#include "ctfapi/CTFTypes.hpp"

namespace Instrument {

	struct task_type_id_t {
	private:
		static std::atomic<ctf_task_type_id_t> _nextTaskTypeId;
	public:
		ctf_task_type_id_t id;

		task_type_id_t()
		{
			id = 0;
		}

		ctf_task_type_id_t autoAssingId()
		{
			id = _nextTaskTypeId++;
			return id;
		}

		bool operator==(task_type_id_t const &other) const
		{
			return id == other.id;
		}
	};
}


#endif // INSTRUMENT_CTF_TASK_TYPE_ID_HPP

