/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_STATISTICS_HPP
#define CPU_STATISTICS_HPP

#include <Chrono.hpp>


class CPUStatistics {

private:

	enum cpu_status_t
	{
		idle_status = 0,
		active_status,
		num_cpu_status
	};

	//! The status in which the CPU currently is
	cpu_status_t _currentStatus;

	//! An array of chronos, one per status
	Chrono _chronos[num_cpu_status];

public:

	inline CPUStatistics() :
		_currentStatus(idle_status)
	{
		// Start this CPU as currently idle
		_chronos[_currentStatus].start();
	}


	inline float getActiveness()
	{
		// Start & stop the current chrono to update the accumulated values
		_chronos[_currentStatus].stop();
		_chronos[_currentStatus].start();

		double idle   = ((double) _chronos[idle_status]);
		double active = ((double) _chronos[active_status]);

		return ( active / (active + idle) );
	}

	inline void cpuBecomesActive()
	{
		_chronos[_currentStatus].stop();
		_currentStatus = active_status;
		_chronos[_currentStatus].start();
	}

	inline void cpuBecomesIdle()
	{
		_chronos[_currentStatus].stop();
		_currentStatus = idle_status;
		_chronos[_currentStatus].start();
	}

};

#endif // CPU_STATISTICS_HPP
