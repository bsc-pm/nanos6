/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_SCHEDULER_HPP
#define INSTRUMENT_NULL_SCHEDULER_HPP

#include "../api/InstrumentScheduler.hpp"
#include "InstrumentTaskId.hpp"

namespace Instrument {

	inline void enterAddReadyTask() {}

	inline void exitAddReadyTask() {}

	inline void enterGetReadyTask() {}

	inline void exitGetReadyTask() {}

	inline void enterSchedulerLock() {}

	inline void schedulerLockBecomesServer() {}

	inline void exitSchedulerLockAsClient(
		__attribute__((unused)) task_id_t taskId
	) {
	}

	inline void exitSchedulerLockAsClient() {}

	inline void schedulerLockServesTask(
		__attribute__((unused)) task_id_t taskId
	) {
	}

	inline void exitSchedulerLockAsServer() {}

	inline void exitSchedulerLockAsServer(
		__attribute__((unused)) task_id_t taskId
	) {
	}
}

#endif // INSTRUMENT_NULL_SCHEDULER_HPP
