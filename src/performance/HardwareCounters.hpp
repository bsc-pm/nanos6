/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_HPP
#define HARDWARE_COUNTERS_HPP


#include <cstddef>
#include <string>
#include <utility>


namespace HardwareCounters {
	enum counter_types_t {
		no_counters_counters_type = 0,
		papi_counters_type
	};
	
	enum preset_counter_t {
		real_frequency_counter = 0,
		virtual_frequency_counter,
		ipc_counter,
		l1_miss_ratio_counter,
		l2_miss_ratio_counter,
		l3_miss_ratio_count,
		fpc_counter,
		real_nsecs_counter,
		virtual_nsecs_counter,
		total_instructions,
		total_preset_counter
	};
	
	extern char const * const _presetCounterNames[total_preset_counter];
	
	
	struct counter_value_t {
		bool _isInteger;
		std::string _name;
		union {
			long _integerValue;
			double _floatValue;
		};
		std::string _units;
		
		counter_value_t(std::string const &name, double value, std::string const &units = "")
			: _isInteger(false), _name(name), _floatValue(value), _units(units)
		{
		}
		
		counter_value_t(std::string const &name, long value, std::string const &units = "")
			: _isInteger(true), _name(name), _integerValue(value), _units(units)
		{
		}
	};
	
	
	//! \brief A hardware counters set. For instance (1) a set for user code and (2) a set for the runtime code
	template <int NUM_SETS = 1>
	class Counters;
	
	template <int NUM_SETS = 1>
	class ThreadCounters;
	
	
	void initialize();
	void shutdown();
	
	inline void initializeThread();
	inline void shutdownThread();
	
}


#if HAVE_PAPI
#include "PAPI/PAPIHardwareCounters.hpp"
#else
#include "no-HC/NoHardwareCounters.hpp"
#endif


#endif // HARDWARE_COUNTERS_HPP
