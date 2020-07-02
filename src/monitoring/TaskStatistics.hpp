/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_STATISTICS_HPP
#define TASK_STATISTICS_HPP

#include <atomic>
#include <cassert>

#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"

#include <Chrono.hpp>


#define DEFAULT_COST 1

enum monitoring_task_status_t {
	/* The task is ready to be executed */
	ready_status = 0,
	/* The task is being executed */
	executing_status,
	/* An aggregation of runtime + pending + blocked */
	paused_status,
	num_status,
	null_status = -1
};


class TasktypeStatistics;

class TaskStatistics {

private:

	//! A pointer to the accumulated statistics of this task's tasktype
	TasktypeStatistics *_tasktypeStatistics;

	//! The computational cost of the task
	size_t _cost;

	//! Number of alive children of the task + 1 (+1 due to this task also
	//! being accounted)
	//! When this counter reaches 0, it means timing data can be accumulated
	//! by whoever decreased it to 0 (hence why +1 is needed for the task
	//! itself to avoid early accumulations)
	std::atomic<size_t> _numChildrenAlive;

	/*    GENERAL METRICS    */

	//! The number of tasks created by this one
	size_t _numChildren;

	/*    TIMING METRICS    */

	//! Array of stopwatches to monitor timing for the task in teach status
	Chrono _chronometers[num_status];

	//! Id of the currently active stopwatch (status)
	monitoring_task_status_t _currentChronometer;

	//! Elapsed execution ticks of children tasks (ticks, not time)
	std::atomic<size_t> _childrenTimes[num_status];

	//! Whether the task has a predicted elapsed execution time
	bool _hasPrediction;

	//! Whether a task in the chain of ancestors of this task had a prediction
	bool _ancestorHasPrediction;

	//! The predicted elapsed execution time of the task
	double _timePrediction;

	//! An approximation the time that has been completed by this task already,
	//! only accounting the elapsed time of children tasks when they complete
	//! the execution of their user code.
	//! NOTE: Ticks of children tasks, not converted to time for faster accumulation.
	//! These are used later on to obtain preciser predictions by subtracting this
	//! time from accumulations of cost at the Tasktype level. When the current task
	//! finishes its execution completely, this time has to be decreased from
	//! TasktypeStatistics, hence why we save it here
	std::atomic<size_t> _completedTime;

	/*    HW COUNTER METRICS    */

	//! Whether the taks has predictions for each hardware counter
	bool *_hasCounterPrediction;

	//! Predictions for each hardware counter of the task
	//! NOTE: Predictions of HWCounters must be doubles due to being computed
	//! using normalized values and products
	double *_counterPredictions;

public:

	//! \brief Constructor
	//!
	//! \param[in] allocationAddress The allocation address for dynamically
	//! allocated parameters
	inline TaskStatistics(void *allocationAddress) :
		_tasktypeStatistics(nullptr),
		_cost(DEFAULT_COST),
		_numChildrenAlive(1),
		_numChildren(0),
		_currentChronometer(null_status),
		_hasPrediction(false),
		_ancestorHasPrediction(false),
		_timePrediction(0.0),
		_completedTime(0),
		_hasCounterPrediction(nullptr),
		_counterPredictions(nullptr)
	{
		for (size_t i = 0; i < num_status; ++i) {
			_childrenTimes[i] = 0;
		}

		const size_t numEvents = HardwareCounters::getNumEnabledCounters();
		if (numEvents != 0) {
			assert(allocationAddress != nullptr);
		}

		_hasCounterPrediction = (bool *) allocationAddress;
		_counterPredictions = (double *) ((char *) allocationAddress + (numEvents * sizeof(bool)));
		for (size_t i = 0; i < numEvents; ++i) {
			_hasCounterPrediction[i] = false;
			_counterPredictions[i] = 0.0;
		}
	}

	inline void reinitialize()
	{
		_tasktypeStatistics = nullptr;
		_cost = DEFAULT_COST;
		_numChildrenAlive = 1;
		_numChildren = 0;
		_currentChronometer = null_status;
		_hasPrediction = false;
		_ancestorHasPrediction = false;
		_timePrediction = 0.0;
		_completedTime = 0;

		for (size_t i = 0; i < num_status; ++i) {
			_chronometers[i].restart();
			_childrenTimes[i] = 0;
		}

		const size_t numEvents = HardwareCounters::getNumEnabledCounters();
		for (size_t i = 0; i < numEvents; ++i) {
			_hasCounterPrediction[i] = false;
			_counterPredictions[i] = 0.0;
		}
	}

	/*    SETTERS & GETTERS    */

	inline void setTasktypeStatistics(TasktypeStatistics *tasktypeStatistics)
	{
		_tasktypeStatistics = tasktypeStatistics;
	}

	inline TasktypeStatistics *getTasktypeStatistics() const
	{
		return _tasktypeStatistics;
	}

	inline void setCost(size_t cost)
	{
		_cost = cost;
	}

	inline size_t getCost() const
	{
		return _cost;
	}

	inline void increaseNumChildrenAlive()
	{
		++_numChildrenAlive;
	}

	// NOTE: Unnecessary, use markAsFinished instead
	// inline bool decreaseNumChildrenAlive();

	//! \brief Mark this task as finished, decreasing the number of children alive
	//!
	//! \return Whether there are no more children alive
	inline bool markAsFinished()
	{
		assert(_numChildrenAlive.load() > 0);

		int aliveChildren = (--_numChildrenAlive);
		return (aliveChildren == 0);
	}

	inline size_t getNumChildrenAlive() const
	{
		return _numChildrenAlive.load();
	}

	//! \brief Increase the number of tasks created by this one
	inline void increaseNumChildren()
	{
		++_numChildren;
	}

	//! \brief Get the number of children tasks created by this one
	inline size_t getNumChildrenTasks() const
	{
		return _numChildren;
	}

	//! \brief Get a representation of the elapsed ticks of a timer
	//!
	//! \param[in] id The timer's timing status id
	//!
	//! \return The elapsed ticks of the timer
	inline size_t getChronoTicks(monitoring_task_status_t id) const
	{
		assert(id < num_status);

		return _chronometers[id].getAccumulated();
	}

	inline size_t getChildrenTimes(monitoring_task_status_t id)
	{
		assert(id < num_status);

		return _childrenTimes[id].load();
	}

	inline void setHasPrediction(bool hasPrediction)
	{
		_hasPrediction = hasPrediction;
	}

	inline bool hasPrediction() const
	{
		return _hasPrediction;
	}

	inline void setAncestorHasPrediction(bool ancestorHasPrediction)
	{
		_ancestorHasPrediction = ancestorHasPrediction;
	}

	inline bool ancestorHasPrediction() const
	{
		return _ancestorHasPrediction;
	}

	inline void setTimePrediction(double prediction)
	{
		_timePrediction = prediction;
	}

	inline double getTimePrediction() const
	{
		return _timePrediction;
	}

	inline void increaseCompletedTime(size_t elapsed)
	{
		_completedTime += elapsed;
	}

	inline size_t getCompletedTime()
	{
		return _completedTime.load();
	}

	inline void setHasCounterPrediction(size_t counterId, bool hasPrediction)
	{
		_hasCounterPrediction[counterId] = hasPrediction;
	}

	inline bool hasCounterPrediction(size_t counterId) const
	{
		return _hasCounterPrediction[counterId];
	}

	inline void setCounterPrediction(size_t counterId, double value)
	{
		_counterPredictions[counterId] = value;
	}

	inline double getCounterPrediction(size_t counterId) const
	{
		return _counterPredictions[counterId];
	}

	/*    TIMING-RELATED METHODS    */

	//! \brief Start/resume a chrono. If resumed, the active chrono must pause
	//!
	//! \param[in] id the timing status of the stopwatch to start/resume
	//!
	//! \return The previous timing status of the task
	inline monitoring_task_status_t startTiming(monitoring_task_status_t id)
	{
		assert(id < num_status);

		// Change the current timing status
		const monitoring_task_status_t oldId = _currentChronometer;
		_currentChronometer = id;

		// Resume the next chrono
		if (oldId == null_status) {
			_chronometers[_currentChronometer].start();
		} else {
			_chronometers[oldId].continueAt(_chronometers[_currentChronometer]);
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
		const monitoring_task_status_t oldId = _currentChronometer;

		if (_currentChronometer != null_status) {
			_chronometers[_currentChronometer].stop();
		}
		_currentChronometer = null_status;

		return oldId;
	}

	//! \brief Get the elapsed execution time of the task
	inline double getElapsedExecutionTime() const
	{
		// First convert children ticks into Chronos to obtain elapsed time
		Chrono executionTimer(_childrenTimes[executing_status].load());

		// Return the aggregation of timing of the task plus its child tasks
		return ((double) _chronometers[executing_status]) + (double) executionTimer;
	}

	//! \brief Accumulate children statistics into the current task
	//!
	//! \param[in] childChronos An array of stopwatches that contain timing
	//! data of the execution of a children task
	//! \param[in] childTimes Accumulated elapsed ticks (one per timing status)
	//! of children tasks created by a child task of the current one
	inline void accumulateChildrenStatistics(TaskStatistics *childStatistics)
	{
		assert(childStatistics != nullptr);

		for (short i = 0; i < num_status; ++i) {
			_childrenTimes[i] +=
				childStatistics->getChronoTicks((monitoring_task_status_t) i) +
				childStatistics->getChildrenTimes((monitoring_task_status_t) i);
		}
	}

	//! \brief Get the size of dynamically allocated parameters
	static inline size_t getTaskStatisticsSize()
	{
		const size_t numEvents = HardwareCounters::getNumEnabledCounters();
		return (numEvents * (sizeof(double) + sizeof(bool)));
	}

};

#endif // TASK_STATISTICS_HPP
