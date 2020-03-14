/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_ID_HPP
#define INSTRUMENT_CTF_TASK_ID_HPP

#include <nanos6.h>
#include <atomic>
#include <stdint.h>

namespace Instrument {
	//! This is the default task identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.

	extern std::atomic<uint32_t> _nextTaskId;

	struct CTFTaskInfo {
		nanos6_task_info_t *_nanos6TaskInfo;
		uint32_t _taskId;
		long _priority;

		CTFTaskInfo(nanos6_task_info_t *nanos6TaskInfo)
			: _nanos6TaskInfo(nanos6TaskInfo), _priority(0)
		{
			_taskId = _nextTaskId++;
		}
	};

	struct task_id_t {
		CTFTaskInfo *_ctfTaskInfo;

		task_id_t()
			: _ctfTaskInfo(nullptr)
		{
		}

		task_id_t(CTFTaskInfo *ctfTaskInfo)
			: _ctfTaskInfo(ctfTaskInfo)
		{
		}
	};
}

#endif // INSTRUMENT_CTF_TASK_ID_HPP
