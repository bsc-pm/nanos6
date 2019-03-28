#ifndef TASK_STATISTICS_HPP
#define TASK_STATISTICS_HPP

#include <string>

#include "lowlevel/SpinLock.hpp"

#include <Chrono.hpp>


#define DEFAULT_COST 1

enum monitoring_task_status_t {
	pending_status = 0,
	ready_status,
	executing_status,
	blocked_status,
	runtime_status,
	num_status,
	null_status = -1
};

char const * const statusDescriptions[num_status] = {
	"Pending Status",
	"Ready Status",
	"Executing Status",
	"Blocked Status",
	"Runtime Status"
};


class TaskStatistics {

private:
	
	//! A string that identifies the tasktype
	std::string _label;
	
	//! The computational cost of the task
	size_t _cost;
	
	//! Array of stopwatches to monitor timing for the task
	Chrono _chronos[num_status];
	
	//! Id of the currently active stopwatch (status)
	monitoring_task_status_t _currentId;
	
	
public:
	
	inline TaskStatistics() :
		_cost(DEFAULT_COST),
		_currentId(null_status)
	{
	}
	
	
	//    SETTERS & GETTERS    //
	
	inline void setLabel(const std::string &label)
	{
		_label = label;
	}
	
	inline const std::string &getLabel() const
	{
		return _label;
	}
	
	inline void setCost(size_t cost)
	{
		_cost = cost;
	}
	
	inline size_t getCost() const
	{
		return _cost;
	}
	
	inline const Chrono *getChronos() const
	{
		return _chronos;
	}
	
	//! \brief Get a representation of the elapsed ticks of a timer
	//! \param id The timer's timing status id
	inline size_t getChronoTicks(monitoring_task_status_t id) const
	{
		return _chronos[id].getAccumulated();
	}
	
	//! \brief Get the elapsed execution time of a timer (in microseconds)
	//! \param id The timer's timing status id
	inline double getElapsedTiming(monitoring_task_status_t id) const
	{
		return ((double) _chronos[id]);
	}
	
	inline monitoring_task_status_t getCurrentTimingStatus() const
	{
		return _currentId;
	}
	
	
	//    TIMING-RELATED    //
	
	//! \brief Start/resume a chrono. If resumed, the active chrono must pause
	//! \param[in] id the timing status of the stopwatch to start/resume
	//! \return The previous status of the task
	inline monitoring_task_status_t startTiming(monitoring_task_status_t id)
	{
		// Change the current timing status
		const monitoring_task_status_t oldId = _currentId;
		_currentId = id;
		
		// Resume the next chrono
		if (oldId == null_status) {
			_chronos[_currentId].start();
		}
		else {
			_chronos[oldId].continueAt(_chronos[_currentId]);
		}
		
		return oldId;
	}
	
	//! \brief Stop/pause a chrono
	//! \param[in] id the timing status of the stopwatch to stop/pause
	//! \return The previous timing status of the task
	inline monitoring_task_status_t stopTiming()
	{
		const monitoring_task_status_t oldId = _currentId;
		
		if (_currentId != null_status) {
			_chronos[_currentId].stop();
		}
		_currentId = null_status;
		
		return oldId;
	}
	
	//! \brief Get the elapsed execution time of the task
	inline double getElapsedTime() const
	{
		return getElapsedTiming(executing_status) +
			getElapsedTiming(runtime_status);
	}
	
};

#endif // TASK_STATISTICS_HPP
