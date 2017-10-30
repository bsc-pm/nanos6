/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_LINK_HPP
#define DATA_ACCESS_LINK_HPP


#include "DataAccessObjectType.hpp"


class Task;


struct DataAccessLink {
	Task *_task;
	DataAccessObjectType _objectType;
	
	DataAccessLink()
		: _task(nullptr)
	{
	}
	
	DataAccessLink(Task *task, DataAccessObjectType objectType)
		: _task(task), _objectType(objectType)
	{
	}
};


#endif // DATA_ACCESS_LINK_HPP
