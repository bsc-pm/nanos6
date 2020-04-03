/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef LOCAL_SCHEDULER_HPP
#define LOCAL_SCHEDULER_HPP

#include "SchedulerInterface.hpp"

class LocalScheduler : public SchedulerInterface {
public:
	~LocalScheduler()
	{}

	inline std::string getName() const
	{
		return "LocalScheduler";
	}
};

#endif // LOCAL_SCHEDULER_HPP
