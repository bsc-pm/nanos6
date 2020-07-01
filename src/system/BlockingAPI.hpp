/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef BLOCKING_API_HPP
#define BLOCKING_API_HPP

#include "tasks/Task.hpp"

namespace BlockingAPI {

	//! \brief Block the current task. The task remains blocked until it is explicitly unblocked
	//!
	//! \param fromUserCode Whether this function is called form user code
	//! or runtime code
	void blockCurrentTask(bool fromUserCode = false);

	//! \brief Unblocks a previously blocked task
	//!
	//! \param task The task about to be unblocked
	//! \param fromUserCode Whether this function is called form user code
	//! or runtime code
	void unblockTask(Task *task, bool fromUserCode = false);

}

#endif // BLOCKING_API_HPP
