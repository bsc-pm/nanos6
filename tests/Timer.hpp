/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TIMER_HPP
#define TIMER_HPP


#include <cstdlib>

#include <sys/time.h>

class Timer {
	double _startTime;
	double _endTime;
	double _accumulated;
	
	inline double getusec()
	{
		struct timeval currentTime;
		
		gettimeofday(&currentTime, 0);
		
		return ( (double)currentTime.tv_sec * (double)1e6 ) + (double)currentTime.tv_usec ;
	}
	
public:
	Timer()
		: _startTime(0.0), _endTime(0.0), _accumulated(0.0)
	{
		start();
	}
	
	inline void start()
	{
		_startTime = getusec();
	}
	
	inline void stop()
	{
		_endTime = getusec();
		_accumulated += (_endTime - _startTime);
	}
	
	inline void reset()
	{
		_startTime = 0.0;
		_endTime = 0.0;
		_accumulated = 0.0;
	}
	
	inline bool hasBeenStartedAtLeastOnce()
	{
		return (_startTime != 0.0);
	}
	
	inline bool hasBeenStoppedAtLeastOnce()
	{
		return (_accumulated != 0.0);
	}
	
	inline double lap()
	{
		stop();
		start();
		return _accumulated;
	}
	
	inline double lapAndReset()
	{
		stop();
		double current = _accumulated;
		reset();
		start();
		return current;
	}
	
	inline operator double()
	{
		return _accumulated;
	}
	
	inline operator long int()
	{
		return _accumulated;
	}
	
};


#endif // TIMER_HPP
