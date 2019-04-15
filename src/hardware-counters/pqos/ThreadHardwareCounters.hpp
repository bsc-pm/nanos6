/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_THREAD_HARDWARE_COUNTERS_HPP
#define PQOS_THREAD_HARDWARE_COUNTERS_HPP

#include <pqos.h>


class ThreadHardwareCounters {

private:
	
	//! Thread id
	pid_t _tid;
	
	//! PQoS-events related data
	pqos_mon_data *_data;
	
	
public:
	
	inline ThreadHardwareCounters()
	{
	}
	
	
	inline void setTid(pid_t tid)
	{
		_tid = tid;
	}
	
	inline pid_t getTid() const
	{
		return _tid;
	}
	
	inline void setData(pqos_mon_data *data)
	{
		_data = data;
	}
	
	inline pqos_mon_data *getData() const
	{
		return _data;
	}
	
};

#endif // PQOS_THREAD_HARDWARE_COUNTERS_HPP
