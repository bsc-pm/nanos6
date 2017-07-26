/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TIMER_HPP
#define TIMER_HPP

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>

#include <string.h>
#include <time.h>


namespace Instrument {

class Timer {
	struct InternalRepresentation {
		struct timespec _ts;
		
		InternalRepresentation()
		{
			_ts.tv_sec = 0;
			_ts.tv_nsec = 0;
		}
		
		operator timespec()
		{
			return _ts;
		}
		
		operator const timespec() const
		{
			return _ts;
		}
		
		InternalRepresentation &operator+=(struct timespec const &ts)
		{
			#ifndef NDEBUG
				InternalRepresentation initial = *this;
			#endif
			
			_ts.tv_sec += ts.tv_sec;
			_ts.tv_nsec += ts.tv_nsec;
			
			long extra_sec = _ts.tv_nsec / 1000000000L;
			_ts.tv_sec += extra_sec;
			_ts.tv_nsec -= extra_sec * 1000000000L;
			
			#ifndef NDEBUG
				assert(*this >= initial);
			#endif
			
			return *this;
		}
		
		InternalRepresentation operator-(struct timespec const &ts)
		{
			assert(*this >= ts);
			
			InternalRepresentation result;
			
			if (_ts.tv_nsec < ts.tv_nsec) {
				long nsec = 1000000000L + _ts.tv_nsec - ts.tv_nsec;
				result._ts.tv_nsec = nsec;
				result._ts.tv_sec = _ts.tv_sec - 1 - ts.tv_sec;
			} else {
				result._ts.tv_nsec = _ts.tv_nsec - ts.tv_nsec;
				result._ts.tv_sec = _ts.tv_sec - ts.tv_sec;
			}
			
			assert(result <= *this);
			
			return result;
		}
		
		InternalRepresentation &operator-=(struct timespec const &ts)
		{
			assert(*this >= ts);
			
			#ifndef NDEBUG
				InternalRepresentation initial = *this;
			#endif
			
			if (_ts.tv_nsec < ts.tv_nsec) {
				long nsec = 1000000000L + _ts.tv_nsec - ts.tv_nsec;
				_ts.tv_nsec = nsec;
				_ts.tv_sec = _ts.tv_sec - 1 - ts.tv_sec;
			} else {
				_ts.tv_nsec = _ts.tv_nsec - ts.tv_nsec;
				_ts.tv_sec = _ts.tv_sec - ts.tv_sec;
			}
			
			#ifndef NDEBUG
				assert(*this <= initial);
			#endif
			
			return *this;
		}
		
		inline InternalRepresentation &operator/=(long divisor)
		{
			#ifndef NDEBUG
				InternalRepresentation initial = *this;
			#endif
			
			double sec_remainder = _ts.tv_sec % divisor;
			_ts.tv_sec /= divisor;
			_ts.tv_nsec /= divisor;
			
			double extra_nsec = sec_remainder * (double) 1000000000L;
			extra_nsec /= ((double) divisor);
			
			_ts.tv_nsec += extra_nsec;
			
			#ifndef NDEBUG
				assert(*this <= initial);
			#endif
			
			return *this;
		}
		
		bool operator>(struct timespec const &other) const
		{
			if (_ts.tv_sec > other.tv_sec) {
				return true;
			} else if (_ts.tv_sec == other.tv_sec) {
				return (_ts.tv_nsec > other.tv_nsec);
			} else {
				return false;
			}
		}
		
		bool operator>=(struct timespec const &other) const
		{
			if (_ts.tv_sec > other.tv_sec) {
				return true;
			} else if (_ts.tv_sec == other.tv_sec) {
				return (_ts.tv_nsec >= other.tv_nsec);
			} else {
				return false;
			}
		}
		
		bool operator<(struct timespec const &other) const
		{
			if (_ts.tv_sec < other.tv_sec) {
				return true;
			} else if (_ts.tv_sec == other.tv_sec) {
				return (_ts.tv_nsec < other.tv_nsec);
			} else {
				return false;
			}
		}
		
		bool operator<=(struct timespec const &other) const
		{
			if (_ts.tv_sec < other.tv_sec) {
				return true;
			} else if (_ts.tv_sec == other.tv_sec) {
				return (_ts.tv_nsec <= other.tv_nsec);
			} else {
				return false;
			}
		}
		
		bool operator==(struct timespec const &other) const
		{
			return (_ts.tv_sec == other.tv_sec) && (_ts.tv_nsec == other.tv_nsec);
		}
		
		bool operator!=(struct timespec const &other) const
		{
			return (_ts.tv_sec != other.tv_sec) || (_ts.tv_nsec != other.tv_nsec);
		}
		
		double veryExplicitConversionToDouble() const
		{
			double result = _ts.tv_sec;
			result *= 1000000000.0;
			result += _ts.tv_nsec;
			return result;
		}
		
		long veryExplicitConversionToLong() const
		{
			return (_ts.tv_sec * 1000000000L) + _ts.tv_nsec;
		}
		
	};
	
	
	InternalRepresentation _startTime;
	InternalRepresentation _endTime;
	InternalRepresentation _accumulated;
	
	inline void getTime(InternalRepresentation &t)
	{
		int rc = clock_gettime(CLOCK_MONOTONIC, &t._ts);
		if (rc != 0) {
			int error = errno;
			std::cerr << "Error reading time: " << strerror(error) << std::endl;
			exit(1);
		}
	}
	
	inline void accumulateLast()
	{
		assert(_startTime != InternalRepresentation());
		assert(_endTime >= _startTime);
		_accumulated += _endTime - _startTime;
	}
	
	
public:
	Timer(bool startCounting = false)
		: _startTime(), _endTime(), _accumulated()
	{
		if (startCounting) {
			start();
		}
	}
	
	static std::string getUnits()
	{
		return std::string("ns");
	}
	
	inline void start()
	{
		getTime(_startTime);
	}
	
	inline void stop()
	{
		getTime(_endTime);
		accumulateLast();
	}
	
	inline void inheritStopTime(Timer const &reference)
	{
		_endTime = reference._endTime;
		accumulateLast();
	}
	
	inline void fixStopTimeFrom(Timer const &reference)
	{
		if (isRunning()) {
			if (reference._endTime < _startTime) {
				// Too late, the accumulated time may include late periods
			} else {
				// Stop at the reference time
				_endTime = reference._endTime;
				accumulateLast();
			}
		} else {
			if (reference._endTime < _startTime) {
				// Too late, the accumulated time may include late periods
				// But at least we can discount the last one
				_accumulated -= _endTime - _startTime;
			} else if (reference._endTime < _endTime) {
				// Discount the fraction outside the reference end time
				_accumulated -= _endTime - reference._endTime;
			} else {
				// Nothing to fix
			}
		}
	}
	
	inline void continueAt(Timer &other)
	{
		stop();
		other._startTime = _endTime;
	}
	
	inline void reset()
	{
		_startTime = InternalRepresentation();
		_endTime = InternalRepresentation();
		_accumulated = InternalRepresentation();
	}
	
	inline bool empty() const
	{
		return
			(_startTime == InternalRepresentation())
			&& (_endTime == InternalRepresentation())
			&& (_accumulated == InternalRepresentation());
	}
	
	bool isRunning() const
	{
		return _startTime > _endTime;
	}
	
	inline bool hasBeenStartedAtLeastOnce()
	{
		return (_startTime != InternalRepresentation());
	}
	
	inline bool hasBeenStoppedAtLeastOnce()
	{
		return (_accumulated != InternalRepresentation());
	}
	
	inline double lap()
	{
		stop();
		start();
		return (double) (*this);
	}
	
	inline double lapAndReset()
	{
		stop();
		double current = (double) (*this);
		reset();
		start();
		return current;
	}
	
	inline operator double() const
	{
		return _accumulated.veryExplicitConversionToDouble();
	}
	
	inline operator long int() const
	{
		return _accumulated.veryExplicitConversionToLong();
	}
	
	inline Timer &operator+=(Timer const &other)
	{
		_accumulated += other._accumulated;
		return *this;
	}
	
	template <typename T>
	inline Timer operator/(T divisor)
	{
		Timer result(*this);
		
		result /= divisor;
		
		return result;
	}
	
	inline Timer &operator/=(long divisor)
	{
		_accumulated /= divisor;
		return *this;
	}
	
	inline std::string fullStatusToString() const
	{
		std::ostringstream oss;
		oss << "Start: " << _startTime.veryExplicitConversionToLong() << ", End: " << _endTime.veryExplicitConversionToLong() << ", Acc: " << _accumulated.veryExplicitConversionToLong();
		return oss.str();
	}
	
	// friend std::ostream &::operator<<(std::ostream &os, Timer const &timer);
};

} // namespace Instrument

inline std::ostream &operator<<(std::ostream &os, Instrument::Timer const &timer)
{
	os << (long int) timer;
	return os;
}


#endif // TIMER_HPP
