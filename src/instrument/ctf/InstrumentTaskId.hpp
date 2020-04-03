/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_ID_HPP
#define INSTRUMENT_CTF_TASK_ID_HPP

#include <nanos6.h>
#include <atomic>
#include <stdint.h>
#include <map>

#include <lowlevel/SpinLock.hpp>

namespace Instrument {
	//! This is the default task identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	typedef std::map <nanos6_task_info_t *, uint16_t>  taskLabelMap_t;
	typedef std::pair<taskLabelMap_t::iterator, bool>  taskLabelMapEntry_t;

	extern std::atomic<uint32_t> _nextTaskId;
	extern uint32_t _nextTaskTypeId;
	extern taskLabelMap_t globalTaskLabelMap;
	extern SpinLock globalTaskLabelLock;

	struct task_id_t {
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

	static inline uint16_t getNewTaskTypeId() {
		return _nextTaskTypeId++;
	}
}

#endif // INSTRUMENT_CTF_TASK_ID_HPP
