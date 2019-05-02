/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_HPP
#define INSTRUMENT_STATS_HPP

#include <list>
#include <map>
#include <vector>

#include <nanos6.h>

#include "Timer.hpp"
#include "instrument/stats/InstrumentHardwareCounters.hpp"
#include "lowlevel/RWTicketSpinLock.hpp"
#include "lowlevel/SpinLock.hpp"


namespace Instrument {
	namespace Stats {
		extern RWTicketSpinLock _phasesSpinLock;
		extern int _currentPhase;
		extern std::vector<Timer> _phaseTimes;
		
		struct TaskTimes {
			Timer _instantiationTime;
			Timer _pendingTime;
			Timer _readyTime;
			Timer _executionTime;
			Timer _blockedTime;
			Timer _zombieTime;
			
			TaskTimes(bool summary)
				: _instantiationTime(!summary),
				_pendingTime(false),
				_readyTime(false),
				_executionTime(false),
				_blockedTime(false),
				_zombieTime(false)
			{
			}
			
			TaskTimes &operator+=(TaskTimes const &instanceTimes)
			{
				_instantiationTime += instanceTimes._instantiationTime;
				_pendingTime += instanceTimes._pendingTime;
				_readyTime += instanceTimes._readyTime;
				_executionTime += instanceTimes._executionTime;
				_blockedTime += instanceTimes._blockedTime;
				_zombieTime += instanceTimes._zombieTime;
				
				return *this;
			}
			
			template <typename T>
			TaskTimes operator/(T divisor) const
			{
				TaskTimes result(*this);
				
				result._instantiationTime /= divisor;
				result._pendingTime /= divisor;
				result._readyTime /= divisor;
				result._executionTime /= divisor;
				result._blockedTime /= divisor;
				result._zombieTime /= divisor;
				
				return result;
			}
			
			Timer getTotal() const
			{
				Timer result;
				
				result += _instantiationTime;
				result += _pendingTime;
				result += _readyTime;
				result += _executionTime;
				result += _blockedTime;
				result += _zombieTime;
				
				return result;
			}
		};
		
		
		struct TaskInfo {
			long _numInstances;
			TaskTimes _times;
			InstrumentHardwareCounters::Counters<> _hardwareCounters;
			
			TaskInfo()
				: _numInstances(0), _times(true), _hardwareCounters()
			{
			}
			
			TaskInfo &operator+=(TaskTimes const &instanceTimes)
			{
				_numInstances++;
				_times += instanceTimes;
				
				return *this;
			}
			
			TaskInfo &operator+=(InstrumentHardwareCounters::Counters<> const &instanceHardwareCounters)
			{
				_hardwareCounters += instanceHardwareCounters;
				
				return *this;
			}
			
			TaskInfo &operator+=(TaskInfo const &other)
			{
				_numInstances += other._numInstances;
				_times += other._times;
				_hardwareCounters += other._hardwareCounters;
				
				return *this;
			}
			
			TaskTimes getMean() const
			{
				return _times / _numInstances;
			}
		};
		
		struct TaskTypeAndTimes {
			nanos6_task_info_t const *_type;
			TaskTimes _times;
			bool _hasParent;
			Timer *_currentTimer;
			InstrumentHardwareCounters::ThreadCounters<> _hardwareCounters;
			
			TaskTypeAndTimes(nanos6_task_info_t const *type, bool hasParent)
				: _type(type), _times(false), _hasParent(hasParent), _currentTimer(&_times._instantiationTime), _hardwareCounters()
			{
			}
		};
		
		
		struct PhaseInfo {
			std::map<nanos6_task_info_t const *, TaskInfo> _perTask;
			Timer _runningTime;
			Timer _blockedTime;
			InstrumentHardwareCounters::Counters<> _hardwareCounters;
			
			PhaseInfo(bool active=true)
				: _perTask(),
				_runningTime(active),
				_blockedTime(false),
				_hardwareCounters()
			{
			}
			
			PhaseInfo &operator+=(PhaseInfo const &other)
			{
				for (auto &perTaskEntry : other._perTask) {
					_perTask[perTaskEntry.first] += perTaskEntry.second;
				}
				
				_runningTime += other._runningTime;
				_blockedTime += other._blockedTime;
				_hardwareCounters += other._hardwareCounters;
				
				return *this;
			}
			
			void stopTimers()
			{
				if (_runningTime.isRunning()) {
					assert(!_blockedTime.isRunning());
					_runningTime.stop();
				} else {
					assert(_blockedTime.isRunning());
					_blockedTime.stop();
				}
			}
			
			void stoppedAt(Timer const &reference)
			{
				if (!_runningTime.empty()) {
					_runningTime.fixStopTimeFrom(reference);
				}
				if (!_blockedTime.empty()) {
					_blockedTime.fixStopTimeFrom(reference);
				}
			}
			
			bool isRunning() const
			{
				return _runningTime.isRunning();
			}
		};
		
		
		struct ThreadInfo {
			std::list<PhaseInfo> _phaseInfo;
			
			ThreadInfo(bool active=true)
				: _phaseInfo()
			{
				_phaseInfo.emplace_back(active);
			}
			
			ThreadInfo &operator+=(ThreadInfo const &other)
			{
				unsigned int phases = other._phaseInfo.size();
				
				while (_phaseInfo.size() < phases) {
					_phaseInfo.emplace_back(false);
				}
				
				auto it = _phaseInfo.begin();
				auto otherIt = other._phaseInfo.begin();
				while (otherIt != other._phaseInfo.end()) {
					(*it) += (*otherIt);
					
					it++;
					otherIt++;
				}
				
				return *this;
			}
			
			PhaseInfo &getCurrentPhaseRef()
			{
				Instrument::Stats::_phasesSpinLock.readLock();
				
				assert(_currentPhase == (_phaseTimes.size() - 1));
				int lastStartedPhase = _phaseInfo.size() - 1;
				
				if (lastStartedPhase == -1) {
					// Add the previous phases as empty
					for (int phase = 0; phase < _currentPhase-1; phase++) {
						_phaseInfo.emplace_back(false);
					}
					// Start the new phase
					_phaseInfo.emplace_back(true);
				} else if (lastStartedPhase < _currentPhase) {
					// Fix the stopping time of the last phase
					bool isRunning = _phaseInfo.back().isRunning();
					_phaseInfo.back().stoppedAt(_phaseTimes[lastStartedPhase]);
					
					// Mark any already finished phase that is missing and the current phase as blocked
					for (int phase = lastStartedPhase+1; phase <= _currentPhase; phase++) {
						_phaseInfo.emplace_back(false);
						
						if (isRunning) {
							_phaseInfo.back()._runningTime = _phaseTimes[phase];
						} else {
							_phaseInfo.back()._blockedTime = _phaseTimes[phase];
						}
					}
				}
				
				Instrument::Stats::_phasesSpinLock.readUnlock();
				
				return _phaseInfo.back();
			}
		};
		
		
		extern SpinLock _threadInfoListSpinLock;
		extern std::list<ThreadInfo *> _threadInfoList;
		extern Timer _totalTime;
	}
}


#endif // INSTRUMENT_STATS_HPP
