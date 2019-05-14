/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_STATISTICS_HPP
#define CPU_STATISTICS_HPP

#include <Chrono.hpp>


class CPUStatistics {

private:
	
	enum cpu_status_t
	{
		idle_status = 0,
		active_status
	};
	
	//! The status in which the CPU currently is
	cpu_status_t _status;
	
	//! Holds timing while a CPU is active
	Chrono _active;
	
	//! Holds timing while a CPU is idle
	Chrono _idle;
	
	//! Percentage of time that the CPU is active
	float _activeness;
	
	
public:
	
	inline CPUStatistics() :
		_status(idle_status),
		_active(),
		_idle(),
		_activeness(0.0)
	{
		// Start this CPU as currently idle
		_idle.start();
	}
	
	
	inline float getActiveness() const
	{
		return _activeness;
	}
	
	inline void cpuBecomesActive()
	{
		if (_status != active_status) {
			_idle.stop();
			_active.start();
			_status = active_status;
		}
	}
	
	inline void cpuBecomesIdle()
	{
		if (_status != idle_status) {
			_active.stop();
			_idle.start();
			_status = idle_status;
			
			// Update the activeness of the CPU
			_activeness = ( ((double)_active) / (((double)_active) + ((double)_idle)) );
		}
	}
	
};

#endif // CPU_STATISTICS_HPP
