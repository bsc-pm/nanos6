/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_DATA_HPP
#define TASKTYPE_DATA_HPP

#include "InstrumentTasktypeData.hpp"
#include "monitoring/TasktypeStatistics.hpp"


//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TasktypeData {

private:

	//! Instrumentation identifier for this Tasktype
	Instrument::TasktypeInstrument _instrumentId;

	//! Monitoring-related statistics per tasktype
	TasktypeStatistics _tasktypeStatistics;

public:

	inline TasktypeData() :
		_instrumentId(),
		_tasktypeStatistics()
	{
	}

	inline Instrument::TasktypeInstrument &getInstrumentationId()
	{
		return _instrumentId;
	}

	inline TasktypeStatistics &getTasktypeStatistics()
	{
		return _tasktypeStatistics;
	}

};

#endif // TASKTYPE_DATA_HPP
