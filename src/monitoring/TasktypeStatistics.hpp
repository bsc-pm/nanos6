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

#include "TaskStatistics.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"


#define PREDICTION_UNAVAILABLE -1.0

namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;


class TasktypeStatistics {

private:

	typedef BoostAcc::accumulator_set<
		double,
		BoostAcc::stats<
			BoostAccTag::count,
			BoostAccTag::mean,
			BoostAccTag::rolling_mean,
			BoostAccTag::sum,
			BoostAccTag::variance
	> > metric_rolling_accumulator_t;

	typedef BoostAcc::accumulator_set<
		double,
		BoostAcc::stats<
			BoostAccTag::count,
			BoostAccTag::mean,
			BoostAccTag::sum,
			BoostAccTag::variance
	> > metric_accumulator_t;

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::sum, BoostAccTag::mean> > accumulator_t;

	//! The rolling-window size for accumulators (elements taken into account)
	static EnvironmentVariable<size_t> _rollingWindow;

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
	std::vector<metric_accumulator_t> _counterAccumulator;

	//! A vector of hardware counter accumulators with normalized metrics
	std::vector<metric_rolling_accumulator_t> _normalizedCounterAccumulator;

	//! Spinlock to ensure atomic access within the previous accumulators
	SpinLock _counterAccumulatorLock;

public:

	inline TasktypeStatistics() :
		_accumulatedCost(0),
		_numAccumulatedInstances(0),
		_numPredictionlessInstances(0),
		_completedTime(0),
		_timingAccumulator(BoostAccTag::rolling_window::window_size = _rollingWindow),
		_timingAccuracyAccumulator(),
		_accumulatedTimeAccumulator(),
		_timingAccumulatorLock(),
		_counterAccumulator(HWCounters::TOTAL_NUM_EVENTS),
		_normalizedCounterAccumulator(
			HWCounters::TOTAL_NUM_EVENTS,
			metric_rolling_accumulator_t(BoostAccTag::rolling_window::window_size = _rollingWindow)
		),
		_counterAccumulatorLock()
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
		double stddev = sqrt(BoostAcc::variance(_timingAccumulator));
		_timingAccumulatorLock.unlock();

		return stddev;
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
	inline double getTimingPrediction(size_t cost)
	{
		double predictedTime = PREDICTION_UNAVAILABLE;

		// Try to inferr a prediction
		_timingAccumulatorLock.lock();
		if (BoostAcc::count(_timingAccumulator)) {
			predictedTime = ((double) cost * BoostAcc::rolling_mean(_timingAccumulator));
		}
		_timingAccumulatorLock.unlock();

		return predictedTime;
	}

	//    HARDWARE COUNTER METRICS    //

	//! \brief Insert a metric into the corresponding accumulator
	//!
	//! \param[in] counterType The type of counter
	//! \param[in] value The value of the metric
	inline void insertNormalizedCounter(HWCounters::counters_t counterType, double value)
	{
		_counterAccumulatorLock.lock();
		_normalizedCounterAccumulator[counterType](value);
		_counterAccumulatorLock.unlock();
	}

	//! \brief Retreive, for a certain type of counter, the sum of accumulated
	//! values of all tasks from this type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the sum of accumulated values
	inline double getCounterSum(HWCounters::counters_t counterType)
	{
		_counterAccumulatorLock.lock();
		double sum = BoostAcc::sum(_counterAccumulator[counterType]);
		_counterAccumulatorLock.unlock();

		return sum;
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated values of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the average accumulated value
	inline double getCounterAverage(HWCounters::counters_t counterType)
	{
		_counterAccumulatorLock.lock();
		double avg = BoostAcc::mean(_counterAccumulator[counterType]);
		_counterAccumulatorLock.unlock();

		return avg;
	}

	//! \brief Retreive, for a certain type of counter, the standard deviation
	//! taking into account all the values in the accumulator of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the standard deviation of the counter
	inline double getCounterStddev(HWCounters::counters_t counterType)
	{
		_counterAccumulatorLock.lock();
		double stddev = sqrt(BoostAcc::variance(_counterAccumulator[counterType]));
		_counterAccumulatorLock.unlock();

		return stddev;
	}

	//! \brief Retreive, for a certain type of counter, the amount of values
	//! in the accumulator (i.e., the number of tasks)
	//!
	//! \param[in] counterType The type of counter
	//! \return A size_t with the number of accumulated values
	inline size_t getCounterNumInstances(HWCounters::counters_t counterType)
	{
		_counterAccumulatorLock.lock();
		size_t count = BoostAcc::count(_counterAccumulator[counterType]);
		_counterAccumulatorLock.unlock();

		return count;
	}

	//! \brief Retreive, for a certain type of counter, the average of all
	//! accumulated normalized values of this task type
	//!
	//! \param[in] counterType The type of counter
	//! \return A double with the average accumulated value
	inline double getCounterRollingAverage(HWCounters::counters_t counterType)
	{
		_counterAccumulatorLock.lock();
		double avg = BoostAcc::rolling_mean(_normalizedCounterAccumulator[counterType]);
		_counterAccumulatorLock.unlock();

		return avg;
	}

	//! \brief Get a hardware counter prediction for a task
	//!
	//! \param[in] counterType The hardware counter's id
	//! \param[in] cost The task's computational cost
	inline double getCounterPrediction(HWCounters::counters_t counterType, size_t cost)
	{
		// Check if a prediction can be inferred
		_counterAccumulatorLock.lock();
		double normalizedValue = PREDICTION_UNAVAILABLE;
		if (BoostAcc::count(_normalizedCounterAccumulator[counterType])) {
			normalizedValue = ((double) cost) * BoostAcc::rolling_mean(_normalizedCounterAccumulator[counterType]);
		}
		_counterAccumulatorLock.unlock();

		return normalizedValue;
	}

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
	inline void accumulateStatisticsAndCounters(
		TaskStatistics *taskStatistics,
		TaskHardwareCounters &taskCounters
	) {
		assert(taskStatistics != nullptr);

		double cost = (double) taskStatistics->getCost();

		//    TIMING    //

		// Normalize the execution time using the task's computational cost
		double elapsed = taskStatistics->getElapsedExecutionTime();
		double normalizedTime = elapsed / cost;

		// Compute the accuracy of the prediction if the task had one
		double accuracy = 0.0;
		bool predictionAvailable = (taskStatistics->hasPrediction() && (elapsed > 0));
		if (predictionAvailable) {
			double predicted = taskStatistics->getTimePrediction();
			double error = 100.0 * (std::abs(predicted - elapsed) / std::max(elapsed, predicted));
			accuracy = 100.0 - error;
		}

		// Accumulate the unitary time, the elapsed time to compute effective
		// parallelism metrics, and the accuracy obtained of a previous prediction
		_timingAccumulatorLock.lock();
		_timingAccumulator(normalizedTime);
		_accumulatedTimeAccumulator(elapsed / 1000.0);
		if (predictionAvailable) {
			_timingAccuracyAccumulator(accuracy);
		}
		_timingAccumulatorLock.unlock();

		//    HARDWARE COUNTERS    //

		const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();
		size_t numEnabledCounters = enabledCounters.size();

		// Pre-compute all the needed values before entering the lock
		double counters[numEnabledCounters];
		double normalizedCounters[numEnabledCounters];
		for (size_t id = 0; id < numEnabledCounters; ++id) {
			counters[id] = taskCounters.getAccumulated(enabledCounters[id]);
			normalizedCounters[id] = (counters[id] / (double) cost);
		}

		// Aggregate all the information into the accumulators
		_counterAccumulatorLock.lock();
		for (size_t id = 0; id < numEnabledCounters; ++id) {
			_counterAccumulator[enabledCounters[id]](counters[id]);
			_normalizedCounterAccumulator[enabledCounters[id]](normalizedCounters[id]);
		}
		_counterAccumulatorLock.unlock();
	}

};

#endif // TASKTYPE_STATISTICS_HPP
