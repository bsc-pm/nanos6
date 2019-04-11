/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_HARDWARE_COUNTERS_HPP
#define INSTRUMENT_NULL_HARDWARE_COUNTERS_HPP


#include <cassert>
#include <cstddef>

#include "system/RuntimeInfo.hpp"


namespace InstrumentHardwareCounters {
	enum implementation_type_t {
		counters_type = no_counters_counters_type
	};
	
	
	inline void initialize()
	{
		RuntimeInfo::addEntry("hardware_counters", "Hardware Counters", "disabled");
	}
	
	inline void shutdown()
	{
	}
	
	inline void initializeThread()
	{
	}
	
	inline void shutdownThread()
	{
	}
	
	
	class CounterSetReference {
	public:
		CounterSetReference()
		{
		}
		
		CounterSetReference &operator+=(__attribute__((unused)) CounterSetReference const &other)
		{
			return *this;
		}
		
		CounterSetReference &operator/=(__attribute__((unused)) size_t divider)
		{
			return *this;
		}
		
		
		class iterator {
		protected:
			friend class InstrumentHardwareCounters::CounterSetReference;
			
			iterator()
			{
			}
			
		public:
			counter_value_t operator*() const
			{
				assert("Attempted to read a hardware counter without hardware counter support" == nullptr);
				
				return counter_value_t("", (long) 0);
			}
			
			iterator &operator++()
			{
				return *this;
			}
			
			iterator operator++(int)
			{
				return *this;
			}
			
			bool operator==(__attribute__((unused)) iterator const &other) const
			{
				// There is only the end iterator
				return true;
			}
			
			bool operator!=(__attribute__((unused)) iterator const &other) const
			{
				// There is only the end iterator
				return false;
			}
			
		}; // iterator
		
		
		iterator begin()
		{
			return iterator();
		}
		
		iterator end()
		{
			return iterator();
		}
	};
	
	
	template <int NUM_SETS>
	class Counters {
	public:
		Counters()
		{
		}
		
		~Counters()
		{
		}
		
		CounterSetReference operator[](__attribute__((unused)) int set)
		{
			return CounterSetReference();
		}
		
		Counters &operator+=(__attribute__((unused)) Counters const &other)
		{
			return *this;
		}
	};
	
	
	template <int NUM_SETS>
	class ThreadCounters : public Counters<NUM_SETS> {
	public:
		ThreadCounters()
			: Counters<NUM_SETS>()
		{
		}
		
		//! \brief Start the hardware counters
		inline bool start()
		{
			return true;
		}
		
		//! \brief Get the current counter values, accumulate them to a set and restart counting from zero
		inline bool accumulateAndRestart(__attribute__((unused)) int set = 0)
		{
			return true;
		}
		
		//! \brief Get the current counter values, accumulate them and stop the counters
		inline bool accumulateAndStop(__attribute__((unused)) int set = 0)
		{
			return true;
		}
		
		//! \brief Stop the counters
		inline void stop(__attribute__((unused)) int set = 0)
		{
		}
		
		
	};
	
	
}


#endif // INSTRUMENT_NULL_HARDWARE_COUNTERS_HPP
