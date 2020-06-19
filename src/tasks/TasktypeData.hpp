/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_DATA_HPP
#define TASKTYPE_DATA_HPP

#include "InstrumentTasktypeData.hpp"

//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TasktypeData {

private:

	//! Instrumentation identifier for this Tasktype
	Instrument::TasktypeInstrument _instrumentId;

public:

	inline TasktypeData() :
		_instrumentId()
	{
	}

	inline Instrument::TasktypeInstrument &getInstrumentationId()
	{
		return _instrumentId;
	}
};

#endif // TASKTYPE_DATA_HPP
