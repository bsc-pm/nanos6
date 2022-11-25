/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_DATA_HPP
#define TASKTYPE_DATA_HPP

#include "InstrumentTasktypeData.hpp"


class TasktypeStatistics;

//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TasktypeData {

private:

	//! Instrumentation identifier for this Tasktype
	Instrument::TasktypeInstrument _instrumentId;

	//! Monitoring-related statistics per tasktype
	TasktypeStatistics *_tasktypeStatistics;

public:

	inline TasktypeData() :
		_instrumentId(),
		_tasktypeStatistics(nullptr)
	{
	}

	inline Instrument::TasktypeInstrument &getInstrumentationId()
	{
		return _instrumentId;
	}
	
	inline void setTasktypeStatistics(TasktypeStatistics *tasktypeStatistics)
	{
		_tasktypeStatistics = tasktypeStatistics;
	}

	inline TasktypeStatistics *getTasktypeStatistics()
	{
		return _tasktypeStatistics;
	}

};

#endif // TASKTYPE_DATA_HPP
