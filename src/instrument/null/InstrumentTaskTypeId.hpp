/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_TASK_TYPE_ID_HPP
#define INSTRUMENT_NULL_TASK_TYPE_ID_HPP


namespace Instrument {
	//! This is the default Task Type id identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct task_type_id_t {
		bool operator==(__attribute__((unused)) task_type_id_t const &other) const
		{
			return true;
		}
	};
}


#endif // INSTRUMENT_NULL_TASK_TYPE_ID_HPP

