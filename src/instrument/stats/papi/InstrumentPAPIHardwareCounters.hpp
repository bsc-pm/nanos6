/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PAPI_HARDWARE_COUNTERS_HPP
#define INSTRUMENT_PAPI_HARDWARE_COUNTERS_HPP

#include <cassert>
#include <cstddef>
#include <vector>

// Work around bug in PAPI header
#define ffsll papi_ffsll
#include <papi.h>
#undef ffsll

#include "InstrumentPAPIHardwareCountersThreadLocalData.hpp"
#include "../InstrumentHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


namespace InstrumentHardwareCounters {
	namespace PAPI {
		extern int _initializationCount;
		
		enum cache_counting_strategy_t {
			no_cache_counting_strategy = 0,
			hits_and_misses_cache_counting_strategy,
			accesses_and_hits_cache_counting_strategy,
			accesses_and_misses_cache_counting_strategy,
			total_hits_and_misses_cache_counting_strategy,
			total_accesses_and_hits_cache_counting_strategy,
			total_accesses_and_misses_cache_counting_strategy,
			cache_counting_strategy_entries
		};
		
		extern char const *cache_counting_strategy_descriptions[cache_counting_strategy_entries];
		
		extern std::vector<int> _papiEventCodes;
		
		struct ThreadLocal {
			int _initializationCount;
			int _eventSet;
			
			ThreadLocal()
				: _initializationCount(0), _eventSet(PAPI_NULL)
			{
			}
		};
		
		typedef signed char event_index_t;
		typedef unsigned long long event_value_t;
		
		extern event_index_t _totalEvents;
		extern event_index_t _l1CacheEventIndex;
		extern event_index_t _l2CacheEventIndex;
		extern event_index_t _l3CacheEventIndex;
		extern event_index_t _FPInstructionsEventIndex;
		extern event_index_t _referenceCyclesEventIndex;
		
		extern cache_counting_strategy_t _l1CacheStrategy;
		extern cache_counting_strategy_t _l2CacheStrategy;
		extern cache_counting_strategy_t _l3CacheStrategy;
	}
	
	
	enum implementation_type_t {
		counters_type = papi_counters_type
	};
	
	
	inline void initializeThread()
	{
		assert(PAPI::_initializationCount > 0);
		PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
		
		assert(threadLocal._initializationCount >= 0);
		
		threadLocal._initializationCount++;
		
		if (threadLocal._initializationCount > 1) {
			return;
		}
		
		assert(threadLocal._eventSet == PAPI_NULL);
		
		int rc = PAPI_create_eventset(&threadLocal._eventSet);
		FatalErrorHandler::failIf(rc == PAPI_ENOMEM, "Not enough memory creating PAPI event set");
		FatalErrorHandler::failIf(rc == PAPI_EINVAL, "Invalid parameter creating PAPI event set");
		
		rc = PAPI_add_events(threadLocal._eventSet, PAPI::_papiEventCodes.data(), PAPI::_papiEventCodes.size());
		FatalErrorHandler::failIf(rc != PAPI_OK, "PAPI failed to add events to an event set");
	}
	
	inline void shutdownThread()
	{
		PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
		
		assert(threadLocal._eventSet != PAPI_NULL);
		
		threadLocal._initializationCount--;
		if (threadLocal._initializationCount < 0) {
			return;
		}
		
		long long stopValues[PAPI::_papiEventCodes.size()];
		PAPI_stop(threadLocal._eventSet, stopValues);
		
		int rc = PAPI_cleanup_eventset(threadLocal._eventSet);
		FatalErrorHandler::warnIf(rc != PAPI_OK, "PAPI failed to clean up an event set");
		
		rc = PAPI_destroy_eventset(&threadLocal._eventSet);
		FatalErrorHandler::warnIf(rc != PAPI_OK, "PAPI failed to destroy an event set");
	}
	
	
	class CounterSetReference {
	protected:
		PAPI::event_value_t &_realNsecs;
		PAPI::event_value_t &_virtualNsecs;
		PAPI::event_value_t *_counterSet;
		
	public:
		CounterSetReference(PAPI::event_value_t &realNsecs, PAPI::event_value_t &virtualNsecs, PAPI::event_value_t *counterSet)
			: _realNsecs(realNsecs), _virtualNsecs(virtualNsecs), _counterSet(counterSet)
		{
		}
		
		CounterSetReference &operator+=(CounterSetReference const &other)
		{
			_realNsecs += other._realNsecs;
			_virtualNsecs += other._virtualNsecs;
			
			for (int i = 0; i < PAPI::_totalEvents; i++) {
				_counterSet[i] += other._counterSet[i];
			}
			
			return *this;
		}
		
		CounterSetReference &operator/=(size_t divider)
		{
			_realNsecs /= divider;
			_virtualNsecs /= divider;
			
			for (int i = 0; i < PAPI::_totalEvents; i++) {
				_counterSet[i] /= divider;
			}
			
			return *this;
		}
		
		
		class iterator {
		protected:
			friend class InstrumentHardwareCounters::CounterSetReference;
			
			CounterSetReference const &_counterSetReference;
			int _index;
			
			bool inValidIndex() const
			{
				switch (_index) {
					case real_frequency_counter:
					case virtual_frequency_counter:
					case ipc_counter:
						return true;
					case l1_miss_ratio_counter:
						return ((PAPI::_l1CacheEventIndex != -1) && (PAPI::_l1CacheStrategy != PAPI::no_cache_counting_strategy));
						break;
					case l2_miss_ratio_counter:
						return ((PAPI::_l2CacheEventIndex != -1) && (PAPI::_l2CacheStrategy != PAPI::no_cache_counting_strategy));
						break;
					case l3_miss_ratio_count:
						return ((PAPI::_l3CacheEventIndex != -1) && (PAPI::_l3CacheStrategy != PAPI::no_cache_counting_strategy));
						break;
					case fpc_counter:
						return (PAPI::_FPInstructionsEventIndex != -1);
						break;
					case real_nsecs_counter:
					case virtual_nsecs_counter:
					case total_instructions:
						return true;
						break;
					default:
						return (_index < ((int) total_preset_counter + PAPI::_totalEvents));
				}
				
				return false;
			}
			
			void advanceOnce()
			{
				_index++;
			}
			
			void advanceUntilValidOrEnd()
			{
				while ((_index < total_preset_counter) && !inValidIndex()) {
					advanceOnce();
				}
			}
			
			iterator(CounterSetReference const &counterSetReference, int index = 0)
				: _counterSetReference(counterSetReference), _index(index)
			{
				advanceUntilValidOrEnd();
			}
			
			double get_miss_ratio_from_strategy(PAPI::event_index_t eventIndex, PAPI::cache_counting_strategy_t strategy) const
			{
				assert(eventIndex != -1);
				assert(strategy != PAPI::no_cache_counting_strategy);
				
				switch (strategy) {
					case PAPI::hits_and_misses_cache_counting_strategy:
						return _counterSetReference._counterSet[eventIndex+1]
							/ (double) (_counterSetReference._counterSet[eventIndex] + _counterSetReference._counterSet[eventIndex+1]);
						break;
					case PAPI::accesses_and_hits_cache_counting_strategy:
						return (double) (_counterSetReference._counterSet[eventIndex] - _counterSetReference._counterSet[eventIndex+1])
							/ _counterSetReference._counterSet[eventIndex];
						break;
					case PAPI::accesses_and_misses_cache_counting_strategy:
						return (double) _counterSetReference._counterSet[eventIndex+1]
							/ _counterSetReference._counterSet[eventIndex];
					case PAPI::total_hits_and_misses_cache_counting_strategy:
						return (double) _counterSetReference._counterSet[eventIndex+1]
							/ (_counterSetReference._counterSet[eventIndex] + _counterSetReference._counterSet[eventIndex+1]);
						break;
					case PAPI::total_accesses_and_hits_cache_counting_strategy:
						return (double) (_counterSetReference._counterSet[eventIndex] - _counterSetReference._counterSet[eventIndex+1])
							/ _counterSetReference._counterSet[eventIndex];
						break;
					case PAPI::total_accesses_and_misses_cache_counting_strategy:
						return (double) _counterSetReference._counterSet[eventIndex+1]
							/ _counterSetReference._counterSet[eventIndex];
						break;
					case PAPI::no_cache_counting_strategy:
						break;
					case PAPI::cache_counting_strategy_entries:
						assert("PAPI cache counting strategy = cache_counting_strategy_entries" == nullptr);
						break;
				}
				
				return 0;
			}
			
			
		public:
			counter_value_t operator*() const
			{
				assert(_index < ((int) total_preset_counter + PAPI::_totalEvents));
				assert(_index >= 0);
				
				switch (_index) {
					case real_frequency_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							_counterSetReference._counterSet[0] / (double) _counterSetReference._realNsecs,
							"GHz"
						);
						break;
					case virtual_frequency_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							_counterSetReference._counterSet[0] / (double) _counterSetReference._virtualNsecs,
							"GHz"
						);
						break;
					case ipc_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							_counterSetReference._counterSet[1] / (double) _counterSetReference._counterSet[0]
						);
						break;
					case l1_miss_ratio_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							get_miss_ratio_from_strategy(PAPI::_l1CacheEventIndex, PAPI::_l1CacheStrategy)
						);
						break;
					case l2_miss_ratio_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							get_miss_ratio_from_strategy(PAPI::_l2CacheEventIndex, PAPI::_l2CacheStrategy)
						);
						break;
					case l3_miss_ratio_count:
						return counter_value_t(
							_presetCounterNames[_index],
							get_miss_ratio_from_strategy(PAPI::_l3CacheEventIndex, PAPI::_l3CacheStrategy)
						);
						break;
					case fpc_counter:
						assert(PAPI::_FPInstructionsEventIndex != -1);
						
						return counter_value_t(
							_presetCounterNames[_index],
							_counterSetReference._counterSet[PAPI::_FPInstructionsEventIndex] / (double) _counterSetReference._realNsecs
						);
						break;
					case real_nsecs_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							(long) _counterSetReference._realNsecs,
							"nsecs"
						);
						break;
					case virtual_nsecs_counter:
						return counter_value_t(
							_presetCounterNames[_index],
							(long) _counterSetReference._virtualNsecs,
							"nsecs"
						);
						break;
					case total_instructions:
						return counter_value_t(
							_presetCounterNames[_index],
							(long) _counterSetReference._counterSet[1],
							"instructions"
						);
						break;
					
					default:
					{
						int papiEventIndex = _index - total_preset_counter;
						PAPI_event_info_t papiEventInfo;
						
						int rc = PAPI_get_event_info(PAPI::_papiEventCodes[papiEventIndex], &papiEventInfo);
						if (rc == PAPI_OK) {
							return counter_value_t(
								papiEventInfo.short_descr,
								(long) _counterSetReference._counterSet[papiEventIndex]
							);
						} else {
							return counter_value_t(
								"Unnamed PAPI event counter",
								(long) _counterSetReference._counterSet[papiEventIndex]
							);
						}
					}
				}
				
				return counter_value_t("", (long) 0);
			}
			
			iterator &operator++()
			{
				advanceOnce();
				advanceUntilValidOrEnd();
				
				return *this;
			}
			
			iterator operator++(int)
			{
				int initialIndex = _index;
				
				advanceOnce();
				advanceUntilValidOrEnd();
				
				return iterator(_counterSetReference, initialIndex);
			}
			
			bool operator==(iterator const &other) const
			{
				// FIXME: Compare the _counterSetReference too
				return (_index == other._index);
			}
			
			bool operator!=(iterator const &other) const
			{
				// FIXME: Compare the _counterSetReference too
				return (_index != other._index);
			}
			
		}; // iterator
		
		
		iterator begin()
		{
			return iterator(*this);
		}
		
		iterator end()
		{
			return iterator(*this, ((int) total_preset_counter + PAPI::_totalEvents));
		}
	};
	
	
	template <int NUM_SETS>
	class Counters {
	protected:
		PAPI::event_value_t _realNsecs[NUM_SETS];
		PAPI::event_value_t _virtualNsecs[NUM_SETS];
		PAPI::event_value_t *_counterSets[NUM_SETS];
		
	public:
		Counters()
		{
			for (int set = 0; set < NUM_SETS; set++) {
				_realNsecs[set] = 0;
				_virtualNsecs[set] = 0;
				_counterSets[set] = new PAPI::event_value_t[PAPI::_totalEvents];
				for (int i = 0; i < PAPI::_totalEvents; i++) {
					_counterSets[set][i] = 0;
				}
			}
		}
		
		~Counters()
		{
			for (int set = 0; set < NUM_SETS; set++) {
				delete[] _counterSets[set];
#ifndef NDEBUG
				_counterSets[set] = nullptr;
#endif
			}
		}
		
		Counters(Counters const &other) = delete;
		Counters &operator=(Counters const &other) = delete;
		
		Counters(Counters &&other)
			: _realNsecs(other._realNsecs), _virtualNsecs(other._virtualNsecs),
			_counterSets(std::move(other._counterSets))
		{
			other._counterSets = nullptr;
		}
		
		CounterSetReference operator[](int set)
		{
			assert(set < NUM_SETS);
			return CounterSetReference(_realNsecs[set], _virtualNsecs[set], _counterSets[set]);
		}
		
		Counters &operator+=(Counters const &other)
		{
			for (int set = 0; set < NUM_SETS; set++) {
				_realNsecs[set] += other._realNsecs[set];
				_virtualNsecs[set] += other._virtualNsecs[set];
				for (int i = 0; i < PAPI::_totalEvents; i++) {
					_counterSets[set][i] += other._counterSets[set][i];
				}
			}
			
			return *this;
		}
	};
	
	
	template <int NUM_SETS>
	class ThreadCounters : public Counters<NUM_SETS> {
	protected:
		PAPI::event_value_t _realStart;
		PAPI::event_value_t _virtualStart;
		bool _valid;
		
	public:
		ThreadCounters()
			: Counters<NUM_SETS>(),
			_realStart(~0ULL), _virtualStart(~0ULL),
			_valid(false)
		{
		}
		
		ThreadCounters(ThreadCounters const &other) = delete;
		ThreadCounters &operator=(ThreadCounters const &other) = delete;
		
		//! \brief Start the hardware counters
		inline bool start()
		{
			PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
			
			int rc = PAPI_start(threadLocal._eventSet);
			_valid = (rc == PAPI_OK);
			
			_realStart = PAPI_get_real_nsec();
			_virtualStart = PAPI_get_virt_nsec();
			
			return _valid;
		}
		
		//! \brief Get the current counter values, accumulate them to a set and restart counting from zero
		inline bool accumulateAndRestart(int set = 0)
		{
			if (_valid) {
				PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
				
				int rc = PAPI_accum(threadLocal._eventSet, (long long *) Counters<NUM_SETS>::_counterSets[set]);
				
				PAPI::event_value_t _realStop = PAPI_get_real_nsec();
				PAPI::event_value_t _virtualStop = PAPI_get_virt_nsec();
				
				_valid = (rc == PAPI_OK);
				
				if (!_valid) {
					return false;
				}
				
				Counters<NUM_SETS>::_realNsecs[set] += (_realStop - _realStart);
				Counters<NUM_SETS>::_virtualNsecs[set] += (_virtualStop - _virtualStart);
				
				_realStart = _realStop;
				_virtualStart = _virtualStop;
				
				return true;
			} else {
				return false;
			}
		}
		
		//! \brief Get the current counter values, accumulate them and stop the counters
		inline bool accumulateAndStop(int set = 0)
		{
			if (_valid) {
				PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
				
				int rc = PAPI_stop(threadLocal._eventSet, (long long *) Counters<NUM_SETS>::_counterSets[set]);
				
				PAPI::event_value_t _realStop = PAPI_get_real_nsec();
				PAPI::event_value_t _virtualStop = PAPI_get_virt_nsec();
				
				if (rc != PAPI_OK) {
					return false;
				}
				
				Counters<NUM_SETS>::_realNsecs[set] += (_realStop - _realStart);
				Counters<NUM_SETS>::_virtualNsecs[set] += (_virtualStop - _virtualStart);
				
				_valid = false;
				
				return true;
			} else {
				return false;
			}
		}
		
		//! \brief Stop the counters
		inline void stop(__attribute__((unused)) int set = 0)
		{
			if (_valid) {
				PAPI::ThreadLocal &threadLocal = PAPI::getCurrentThreadHardwareCounters();
				
				PAPI_stop(threadLocal._eventSet, nullptr);
			}
		}
		
		
	};
	
};


#endif // INSTRUMENT_PAPI_HARDWARE_COUNTERS_HPP
