/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_STATISTICS_HPP
#define TASK_STATISTICS_HPP

#include <atomic>
#include <string>

#include <Chrono.hpp>


#define DEFAULT_COST 1

enum monitoring_task_status_t {
	/* The task has dependencies pending to be resolved */
	pending_status = 0,
	/* The task is ready to be executed */
	ready_status,
	/* The task is being executed */
	executing_status,
	/* The task is blocked (taskwait...) */
	blocked_status,
	/* The task is paused and executing runtime code (creating another task...) */
	runtime_status,
	/* An aggregation of runtime + pending + blocked */
	paused_status,
	num_status,
	null_status = -1
};

char const * const statusDescriptions[num_status] = {
	"Pending Status",
	"Ready Status",
	"Executing Status",
	"Blocked Status",
	"Runtime Status",
	"Paused Status"
};

class TasktypePredictions;

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

	//! A pointer to the TaskStatistics of the parent task
	TaskStatistics *_parentStatistics;

	//! Number of alive children of the task + 1 (+1 due to this task also
	//! being accounted)
	//! When this counter reaches 0, it means timing data can be accumulated
	//! by whoever decreased it to 0 (hence why +1 is needed for the task
	//! itself to avoid early accumulations)
	std::atomic<size_t> _aliveChildren;

	//! Elapsed execution ticks of children tasks (not converted to time)
	std::atomic<size_t> _childrenTimes[num_status];

	//! Whether a timing prediction is available for the task
	bool _taskHasPrediction;

	//! Predicted elapsed execution time of the task
	double _timePrediction;

	//! Whether a task in the chain of ancestors of this task had predictions
	bool _ancestorHasPrediction;

	//! Ticks of children tasks, used to obtain more accurate predictions
	//! (not converted to time, simply internal chronometer ticks)
	std::atomic<size_t> _childCompletionTimes;

	//! A pointer to the accumulated predictions of the tasktype
	TasktypePredictions *_typePredictions;

	//! The number of tasks created by this one
	size_t _numChildrenTasks;

public:

	inline TaskStatistics() :
		_cost(DEFAULT_COST),
		_currentId(null_status),
		_aliveChildren(1),
		_taskHasPrediction(false),
		_ancestorHasPrediction(false),
		_childCompletionTimes(0),
		_typePredictions(nullptr),
		_numChildrenTasks(0)
	{
		for (short i = 0; i < num_status; ++i) {
			_childrenTimes[i] = 0;
		}
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
	//!
	//! \param[in] id The timer's timing status id
	//!
	//! \return The elapsed ticks of the timer
	inline size_t getChronoTicks(monitoring_task_status_t id) const
	{
		return _chronos[id].getAccumulated();
	}

	//! \brief Get the elapsed execution time of a timer (in microseconds)
	//!
	//! \param[in] id The timer's timing status id
	//!
	//! \return The elapsed execution time of the timer in microseconds
	inline double getElapsedTiming(monitoring_task_status_t id) const
	{
		return ((double) _chronos[id]);
	}

	inline monitoring_task_status_t getCurrentTimingStatus() const
	{
		return _currentId;
	}

	inline void setParentStatistics(TaskStatistics *parentStatistics)
	{
		_parentStatistics = parentStatistics;
	}

	inline TaskStatistics *getParentStatistics() const
	{
		return _parentStatistics;
	}

	inline const std::atomic<size_t> *getChildTimes() const
	{
		return _childrenTimes;
	}

	inline size_t getChildTiming(monitoring_task_status_t statusId) const
	{
		return _childrenTimes[statusId].load();
	}

	inline void increaseAliveChildren()
	{
		++_aliveChildren;
	}

	//! \brief Decrease the number of alive children
	//!
	//! \return Whether the decreased child was the last child
	inline bool decreaseAliveChildren()
	{
		assert(_aliveChildren.load() > 0);

		int aliveChildren = (--_aliveChildren);
		return (aliveChildren == 0);
	}

	inline size_t getAliveChildren() const
	{
		return _aliveChildren.load();
	}

	//! \brief Mark this task as finished, decreasing the extra unit of alive
	//! children, which is the current task
	//!
	//! \return Whether there are no more children alive
	inline bool markAsFinished()
	{
		assert(_aliveChildren.load() > 0);

		int aliveChildren = (--_aliveChildren);
		return (aliveChildren == 0);
	}

	//! \brief Increase the number of tasks created by this one
	inline void increaseNumChildren(size_t numChildrenTasks)
	{
		++_numChildrenTasks;
	}

	//! \brief Get the number of children tasks created by this one
	inline size_t getNumChildrenTasks() const
	{
		return _numChildrenTasks;
	}


	//    TIMING-RELATED    //

	//! \brief Start/resume a chrono. If resumed, the active chrono must pause
	//!
	//! \param[in] id the timing status of the stopwatch to start/resume
	//!
	//! \return The previous timing status of the task
	inline monitoring_task_status_t startTiming(monitoring_task_status_t id)
	{
		// Change the current timing status
		const monitoring_task_status_t oldId = _currentId;
		_currentId = id;

		// Resume the next chrono
		if (oldId == null_status) {
			_chronos[_currentId].start();
		} else {
			_chronos[oldId].continueAt(_chronos[_currentId]);
		}

		return oldId;
	}

	//! \brief Stop/pause a chrono
	//!
	//! \param[in] id the timing status of the stopwatch to stop/pause
	//!
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
		// First convert children ticks into Chronos to obtain elapsed time
		Chrono executionTimer(getChildTiming(executing_status));
		Chrono runtimeTimer(getChildTiming(runtime_status));

		// Return the aggregation of timing of the task plus its child tasks
		return getElapsedTiming(executing_status) +
			getElapsedTiming(runtime_status)      +
			(double) executionTimer               +
			(double) runtimeTimer;
	}

	//! \brief Accumulate children tasks timing
	//!
	//! \param[in] childChronos An array of stopwatches that contain timing
	//! data of the execution of a children task
	//! \param[in] childTimes Accumulated elapsed ticks (one per timing status)
	//! of children tasks created by a child task of the current one
	inline void accumulateChildTiming(
		const Chrono *childChronos,
		const std::atomic<size_t> *childTimes
	) {
		assert(childChronos != nullptr);
		assert(childTimes != nullptr);

		for (short i = 0; i < num_status; ++i) {
			_childrenTimes[i] +=
				childChronos[i].getAccumulated() +
				childTimes[i].load();
		}
	}

	inline void setHasPrediction(bool hasPrediction)
	{
		_taskHasPrediction = hasPrediction;
	}

	inline bool hasPrediction() const
	{
		return _taskHasPrediction;
	}

	inline void setTimePrediction(double prediction)
	{
		_timePrediction = prediction;
	}

	inline double getTimePrediction() const
	{
		return _timePrediction;
	}

	inline void setAncestorHasPrediction(bool ancestorHasPrediction)
	{
		_ancestorHasPrediction = ancestorHasPrediction;
	}

	inline bool ancestorHasPrediction() const
	{
		return _ancestorHasPrediction;
	}

	inline size_t getChildCompletionTimes() const
	{
		return _childCompletionTimes.load();
	}

	inline void increaseChildCompletionTimes(size_t elapsed)
	{
		_childCompletionTimes += elapsed;
	}

	inline void setTypePredictions(TasktypePredictions *predictions)
	{
		_typePredictions = predictions;
	}

	inline TasktypePredictions *getTypePredictions() const
	{
		return _typePredictions;
	}

};

#endif // TASK_STATISTICS_HPP
