/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_THREAD_HARDWARE_COUNTERS_HPP
#define PQOS_THREAD_HARDWARE_COUNTERS_HPP

#include <pqos.h>

#include "hardware-counters/ThreadHardwareCounters.hpp"


class PQoSThreadHardwareCounters : public ThreadHardwareCounters {

private:

	//! Thread id
	pid_t _tid;

	//! PQoS structures that hold counter values
	pqos_mon_data *_data;

public:

	inline void setTid(pid_t tid)
	{
		_tid = tid;
	}

	inline pid_t getTid() const
	{
		return _tid;
	}

	inline void setData(pqos_mon_data *threadData)
	{
		_data = threadData;
	}

	inline pqos_mon_data *getData() const
	{
		return _data;
	}

};

#endif // PQOS_THREAD_HARDWARE_COUNTERS_HPP
