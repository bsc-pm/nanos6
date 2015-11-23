#ifndef INSTRUMENT_STATS_HPP
#define INSTRUMENT_STATS_HPP

#include <list>
#include <map>

#include "api/nanos6_rt_interface.h"
#include "lowlevel/SpinLock.hpp"
#include "Timer.hpp"



namespace Instrument {
	namespace Stats {
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
			
			TaskInfo()
				: _numInstances(0), _times(true)
			{
			}
			
			TaskInfo &operator+=(TaskTimes const &instanceTimes)
			{
				_numInstances++;
				_times += instanceTimes;
				
				return *this;
			}
			
			TaskInfo &operator+=(TaskInfo const &other)
			{
				_numInstances += other._numInstances;
				_times += other._times;
				
				return *this;
			}
			
			TaskTimes getMean() const
			{
				return _times / _numInstances;
			}
		};
		
		struct TaskTypeAndTimes {
			nanos_task_info const *_type;
			Timer *_currentTimer;
			TaskTimes _times;
			
			TaskTypeAndTimes(nanos_task_info const *type)
				: _type(type), _currentTimer(&_times._instantiationTime), _times(false)
			{
			}
		};
		
		struct ThreadInfo {
			std::map<nanos_task_info const *, TaskInfo> _perTask;
			Timer _runningTime;
			Timer _blockedTime;
			
			ThreadInfo(bool active=true)
				: _perTask(),
				_runningTime(false),
				_blockedTime(active)
			{
			}
			
			ThreadInfo &operator+=(ThreadInfo const &other)
			{
				for (auto perTaskEntry : other._perTask) {
					_perTask[perTaskEntry.first] += perTaskEntry.second;
				}
				
				_runningTime += other._runningTime;
				_blockedTime += other._blockedTime;
				
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
				_runningTime.fixStopTimeFrom(reference);
				_blockedTime.fixStopTimeFrom(reference);
			}
			
		};
		
		
		extern thread_local ThreadInfo *_threadStats;
		
		extern SpinLock _threadInfoListSpinLock;
		extern std::list<ThreadInfo *> _threadInfoList;
		extern Timer _totalTime;
	}
}


#endif // INSTRUMENT_STATS_HPP
