/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef EVENTS_API_HPP
#define EVENTS_API_HPP

#include "tasks/Task.hpp"

namespace EventsAPI {

	//! \brief Increase the current task's events to prevent the release of dependencies
	//!
	//! \param increment The value to be incremented (must be positive or zero)
	void increaseCurrentTaskEvents(unsigned int increment);

	//! \brief Decrease the task's events and release the dependencies if required
	//!
	//! \param task The task to decrease the events
	//! \param decrement The value to be decremented (must be positive or zero)
	void decreaseTaskEvents(Task *task, unsigned int decrement);
}

#endif // EVENTS_API_HPP
