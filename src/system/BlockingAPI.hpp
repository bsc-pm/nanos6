/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef BLOCKING_API_HPP
#define BLOCKING_API_HPP

#include <cstdint>

#include "tasks/Task.hpp"

namespace BlockingAPI {

	//! \brief Block the current task. The task remains blocked until it is explicitly unblocked
	//!
	//! \param fromUserCode Whether this function is called form user code
	//! or runtime code
	void blockCurrentTask(bool fromUserCode = false);

	//! \brief Unblock a previously blocked task
	//!
	//! \param task The task about to be unblocked
	//! \param fromUserCode Whether this function is called form user code
	//! or runtime code
	void unblockTask(Task *task, bool fromUserCode = false);

	//! \brief Pause the current task for an amount of microseconds
	//!
	//! The task is paused for approximately the amount of microseconds
	//! passed as a parameter. The runtime may choose to execute other
	//! tasks within the execution scope of this call
	//!
	//! \param timeUs The time that should be spent while paused in us
	//!
	//! \returns The actual time spent during the pause
	uint64_t waitForUs(uint64_t timeUs);
}

#endif // BLOCKING_API_HPP
