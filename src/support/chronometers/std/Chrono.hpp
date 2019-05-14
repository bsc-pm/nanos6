/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CHRONO_STD_HPP
#define CHRONO_STD_HPP

#include <chrono>


typedef std::chrono::steady_clock::time_point time_point_t;


class Chrono {

private:
	
	time_point_t _chrono;
	
	size_t _accumulated;
	
	
public:
	
	inline Chrono() :
		_accumulated(0)
	{
	}
	
	inline Chrono(size_t ticks) :
		_accumulated(ticks)
	{
	}
	
	
	inline void start()
	{
		_chrono = std::chrono::steady_clock::now();
	}
	
	inline void stop()
	{
		_accumulated += (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - _chrono).count());
	}
	
	inline void restart()
	{
		_accumulated = 0;
	}
	
	inline void continueAt(Chrono &other)
	{
		stop();
		other.start();
	}
	
	//! \brief Returns a representation in microseconds of the accumulated
	//! time gathered by the chronometer
	inline operator double() const
	{
		return ((double) _accumulated);
	}
	
	inline void operator+=(const Chrono& chrono)
	{
		_accumulated += chrono.getAccumulated();
	}
	
	//! \brief Returns the accumulated ticks of this chronometer, not
	//! converted to time
	inline size_t getAccumulated() const
	{
		return _accumulated;
	}
	
};

#endif // CHRONO_STD_HPP
