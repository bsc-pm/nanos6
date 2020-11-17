/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_STATISTICS_HPP
#define TASKTYPE_STATISTICS_HPP

// NOTE: "array_wrapper" must be included before any other boost modules
// Workaround for a missing include in boost 1.64
#include <boost/version.hpp>
#if BOOST_VERSION == 106400
#include <boost/serialization/array_wrapper.hpp>
#endif

#include <atomic>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <cstddef>

#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "lowlevel/SpinLock.hpp"
#include "support/config/ConfigVariable.hpp"

namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;


class TaskStatistics;

class TasktypeStatistics {

private:

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::mean> > accumulator_t;

	typedef BoostAcc::accumulator_set<
		double,
		BoostAcc::stats<
			BoostAccTag::count,
			BoostAccTag::mean,
			BoostAccTag::sum,
			BoostAccTag::variance
	> > metric_accumulator_t;

	typedef BoostAcc::accumulator_set<
		double,
		BoostAcc::stats<
			BoostAccTag::count,
			BoostAccTag::mean,
			BoostAccTag::rolling_mean,
			BoostAccTag::variance
	> > metric_rolling_accumulator_t;


	//! The rolling-window size for accumulators (elements taken into account)
	static ConfigVariable<int> _rollingWindow;

	//    TIMING METRICS    //

	//! Contains the aggregated computational cost ready to be executed of a tasktype
	std::atomic<size_t> _accumulatedCost;

	//! The number of instances contributing to the accumulated cost
	std::atomic<size_t> _numAccumulatedInstances;

	//! The number of currently active task instances of this type that do not have a prediction
	std::atomic<size_t> _numPredictionlessInstances;

	//! Out of the accumulated cost, this variable contains the amount of time already
	//! completed by children tasks of tasks that have not finished executing yet
	std::atomic<size_t> _completedTime;

	//! An accumulator which holds normalized timing measures of tasks
	metric_rolling_accumulator_t _timingAccumulator;

	//! An accumulator which holds accuracy data of timing predictions
	accumulator_t _timingAccuracyAccumulator;

	//! An accumulator which holds the accumulation of all the elapsed time of tasks (in milliseconds)
	accumulator_t _accumulatedTimeAccumulator;

	//! Spinlock to ensure atomic access within the previous accumulators
	SpinLock _timingAccumulatorLock;

	//    HARDWARE COUNTER METRICS    //

	//! A vector of hardware counter accumulators
	std::vector<metric_accumulator_t> _counterAccumulators;

	//! A vector of hardware counter accumulators with normalized metrics
	std::vector<metric_rolling_accumulator_t> _normalizedCounterAccumulators;

	//! A vector of hardware counter accumulators for the accuracy of predictions
	std::vector<accumulator_t> _counterAccuracyAccumulators;

	//! Spinlock to ensure atomic access within the previous accumulators
	SpinLock _counterAccumulatorsLock;

public:

	//! \brief Constructor
	//!
	//! NOTE: We use 'HWC_TOTAL_NUM_EVENTS' as the number of events instead of
	//! doing it dynamically because there shouldn't be too many objects of this
	//! type, and they're created at runtime-initialization, so we cannot know
	//! at that time how many counters we will really use. Nonetheless, we will
	//! only use as many accumulators as enabled counters (numEnabledCounters)
	inline TasktypeStatistics() :
		_accumulatedCost(0),
		_numAccumulatedInstances(0),
		_numPredictionlessInstances(0),
		_completedTime(0),
		_timingAccumulator(BoostAccTag::rolling_window::window_size = _rollingWindow),
		_timingAccuracyAccumulator(),
		_accumulatedTimeAccumulator(),
		_timingAccumulatorLock(),
		_counterAccumulators(HWCounters::HWC_TOTAL_NUM_EVENTS),
		_normalizedCounterAccumulators(
			HWCounters::HWC_TOTAL_NUM_EVENTS,
			metric_rolling_accumulator_t(BoostAccTag::rolling_window::window_size = _rollingWindow)
		),
		_counterAccuracyAccumulators(HWCounters::HWC_TOTAL_NUM_EVENTS),
		_counterAccumulatorsLock()
	{
	}

	inline ~TasktypeStatistics()
	{
		assert(_accumulatedCost.load() == 0);
		assert(_numAccumulatedInstances.load() == 0);
		assert(_numPredictionlessInstances.load() == 0);
		assert(_completedTime.load() == 0);
	}

	inline void increaseAccumulatedCost(size_t cost)
	{
		_accumulatedCost += cost;
	}

	inline void decreaseAccumulatedCost(size_t cost)
	{
		assert(_accumulatedCost.load() >= cost);

		_accumulatedCost -= cost;
	}

	inline size_t getAccumulatedCost()
	{
		return _accumulatedCost.load();
	}

	inline void increaseNumAccumulatedInstances()
	{
		++_numAccumulatedInstances;
	}

	inline void decreaseNumAccumulatedInstances()
	{
		assert(_numAccumulatedInstances.load() > 0);

		--_numAccumulatedInstances;
	}

	inline size_t getNumAccumulatedInstances()
	{
		return _numAccumulatedInstances.load();
	}

	inline void increaseNumPredictionlessInstances()
	{
		++_numPredictionlessInstances;
	}

	inline void decreaseNumPredictionlessInstances()
	{
		assert(_numPredictionlessInstances.load() > 0);

		--_numPredictionlessInstances;
	}

	inline size_t getNumPredictionlessInstances()
	{
		return _numPredictionlessInstances.load();
	}

	inline void increaseCompletedTime(size_t time)
	{
		_completedTime += time;
	}

	inline void decreaseCompletedTime(size_t time)
	{
		assert(_completedTime.load() >= time);

		_completedTime -= time;
	}

	inline size_t getCompletedTime()
	{
		return _completedTime.load();
	}

	inline double getAccumulatedTime()
	{
		_timingAccumulatorLock.lock();
		double time = BoostAcc::sum(_accumulatedTimeAccumulator);
		_timingAccumulatorLock.unlock();

		return time;
	}

	//    TIMING PREDICTIONS    //

	//! \brief Insert a normalized cost value (time per unit of cost)
	//! in the time accumulators
	inline void insertNormalizedTime(double normalizedTime)
	{
		_timingAccumulatorLock.lock();
		_timingAccumulator(normalizedTime);
		_timingAccumulatorLock.unlock();
	}

	//! \brief Get the standard deviation of the normalized unitary cost of this tasktype
	inline double getTimingStddev()
	{
		_timingAccumulatorLock.lock();
		double variance = BoostAcc::variance(_timingAccumulator);
		_timingAccumulatorLock.unlock();

		return sqrt(variance);
	}

	//! \brief Get the number of task instances that accumulated metrics
	inline size_t getTimingNumInstances()
	{
		_timingAccumulatorLock.lock();
		size_t numInstances = BoostAcc::count(_timingAccumulator);
		_timingAccumulatorLock.unlock();

		return numInstances;
	}

	//! \brief Get the average accuracy of timing predictions of this tasktype
	inline double getTimingAccuracy()
	{
		_timingAccumulatorLock.lock();
		double accuracy = BoostAcc::mean(_timingAccuracyAccumulator);
		_timingAccumulatorLock.unlock();

		return accuracy;
	}

	//! \brief Get the average normalized unitary cost of this tasktype
	inline double getTimingRollingAverage()
	{
		_timingAccumulatorLock.lock();
		double average = BoostAcc::rolling_mean(_timingAccumulator);
		_timingAccumulatorLock.unlock();

		return average;
	}

	//! \brief Get a timing prediction for a task
	//!
	//! \param[in] cost The task's computational costs
	double getTimingPrediction(size_t cost);

	//    HARDWARE COUNTER METRICS    //

	//! \brief Insert a metric into the corresponding accumulator
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \param[in] value The value of the metric
	inline void insertNormalizedCounter(size_t counterId, double value)
	{
		_counterAccumulatorsLock.lock();
		_normalizedCounterAccumulators[counterId](value);
		_counterAccumulatorsLock.unlock();
	}

	//! \brief Retreive, for a certain type of counter, the sum of accumulated
	//! values of all tasks from this type
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A double with the sum of accumulated values
	inline double getCounterSum(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		double sum = BoostAcc::sum(_counterAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return sum;
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated values of this task type
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A double with the average accumulated value
	inline double getCounterAverage(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		double avg = BoostAcc::mean(_counterAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return avg;
	}

	//! \brief Retreive, for a certain type of counter, the standard deviation
	//! taking into account all the values in the accumulator of this task type
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A double with the standard deviation of the counter
	inline double getCounterStddev(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		double variance = BoostAcc::variance(_counterAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return sqrt(variance);
	}

	//! \brief Retreive, for a certain type of counter, the amount of values
	//! in the accumulator (i.e., the number of tasks)
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A size_t with the number of accumulated values
	inline size_t getCounterNumInstances(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		size_t count = BoostAcc::count(_counterAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return count;
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated normalized values of this task type
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A double with the average accumulated value
	inline double getCounterRollingAverage(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		double avg = BoostAcc::rolling_mean(_normalizedCounterAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return avg;
	}

	//! \brief Retreive, for a certain type of counter, the average accuracy
	//! of counter predictions of this tasktype
	//!
	//! \param[in] counterId An identifier relative to the number of enabled events
	//! \return A double with the average accuracy
	inline double getCounterAccuracy(size_t counterId)
	{
		_counterAccumulatorsLock.lock();
		double avg = BoostAcc::mean(_counterAccuracyAccumulators[counterId]);
		_counterAccumulatorsLock.unlock();

		return avg;
	}

	//! \brief Get a hardware counter prediction for a task
	//!
	//! \param[in] counterId The hardware counter's id
	//! \param[in] cost The task's computational cost
	double getCounterPrediction(size_t counterId, size_t cost);

	//    SHARED FUNCTIONS: MONITORING + HWC    //

	//! \brief Accumulate a task's timing statisics and counters to inferr
	//! predictions. More specifically, this function:
	//! - Normalizes task metrics with its cost to later insert these normalized
	//!   metrics into accumulators
	//! - If there were predictions for the previous metrics, the accuracy is
	//!   computed and inserted into accuracy accumulators
	//!
	//! \param[in] taskStatistics The task's statistics
	//! \param[in] taskCounters The task's hardware counters
	void accumulateStatisticsAndCounters(
		TaskStatistics *taskStatistics,
		TaskHardwareCounters &taskCounters
	);

};

#endif // TASKTYPE_STATISTICS_HPP
