/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_TYPE_DATA_HPP
#define TASK_TYPE_DATA_HPP


//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TaskTypeData {

private:

	//! This task type's identifier
	short _id;

public:

	inline TaskTypeData(short id) :
		_id(id)
	{
	}

	inline void setId(short id)
	{
		_id = id;
	}

	inline short getId() const
	{
		return _id;
	}

};

#endif // TASK_TYPE_DATA_HPP
