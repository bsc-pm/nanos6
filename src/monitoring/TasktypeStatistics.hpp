/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_STATISTICS_HPP
#define TASKTYPE_STATISTICS_HPP

#include <atomic>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <cstddef>

#include "TaskStatistics.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"


#define PREDICTION_UNAVAILABLE -1.0

namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;


class TasktypeStatistics {

private:

	//! Contains the aggregated computational cost ready to be executed of a tasktype
	std::atomic<size_t> _accumulatedCost;

	//! Out of the accumulated cost, this variable contains the amount of time already
	//! completed by children tasks of tasks that have not finished executing yet
	std::atomic<size_t> _completedTime;

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::rolling_mean, BoostAccTag::variance, BoostAccTag::count> > time_accumulator_t;
	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::sum, BoostAccTag::mean> > accumulator_t;

	//! An accumulator which holds normalized timing measures of tasks
	time_accumulator_t _timeAccumulator;

	//! An accumulator which holds accuracy data of timing predictions
	accumulator_t _accuracyAccumulator;

	//! Spinlock to ensure atomic access within the previous accumulators
	SpinLock _accumulatorLock;

	//! The rolling-window size for accumulators (elements taken into account)
	static EnvironmentVariable<size_t> _rollingWindow;

public:

	inline TasktypeStatistics() :
		_accumulatedCost(0),
		_completedTime(0),
		_timeAccumulator(BoostAccTag::rolling_window::window_size = _rollingWindow),
		_accuracyAccumulator(),
		_accumulatorLock()
	{
	}

	inline ~TasktypeStatistics()
	{
		assert(_accumulatedCost.load() == 0);
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

	//    TIME PREDICTIONS    //

	//! \brief Insert a normalized time value (time per unit of cost)
	//! in the time accumulators
	inline void insertNormalizedTime(double normalizedTime)
	{
		_accumulatorLock.lock();
		_timeAccumulator(normalizedTime);
		_accumulatorLock.unlock();
	}

	//! \brief Get the average normalized unitary cost of this tasktype
	inline double getAverageNormalizedCost()
	{
		_accumulatorLock.lock();
		double average = BoostAcc::rolling_mean(_timeAccumulator);
		_accumulatorLock.unlock();

		return average;
	}

	//! \brief Get the standard deviation of the normalized unitary cost of this tasktype
	inline double getStddevNormalizedCost()
	{
		_accumulatorLock.lock();
		double stddev = sqrt(BoostAcc::variance(_timeAccumulator));
		_accumulatorLock.unlock();

		return stddev;
	}

	//! \brief Get the number of task instances that accumulated metrics
	inline size_t getNumInstances()
	{
		_accumulatorLock.lock();
		size_t numInstances = BoostAcc::count(_timeAccumulator);
		_accumulatorLock.unlock();

		return numInstances;
	}

	//! \brief Get the average accuracy of timing predictions of this tasktype
	inline double getPredictionAccuracy()
	{
		_accumulatorLock.lock();
		double accuracy = BoostAcc::mean(_accuracyAccumulator);
		_accumulatorLock.unlock();

		return accuracy;
	}

	//! \brief Get a timing prediction for a task
	//!
	//! \param[in] cost The task's computational costs
	inline double getTimePrediction(size_t cost)
	{
		double predictedTime = PREDICTION_UNAVAILABLE;

		// Try to inferr a prediction
		_accumulatorLock.lock();
		if (BoostAcc::count(_timeAccumulator)) {
			predictedTime = ((double) cost * BoostAcc::rolling_mean(_timeAccumulator));
		}
		_accumulatorLock.unlock();

		return predictedTime;
	}

	//! \brief Accumulate a task's timing statisics to inferr predictions
	//! More specifically, this function:
	//! - Normalizes the elapsed execution time of the task with its cost
	//!   to later insert it into the accumulators
	//! - If there was a timing prediction for the task, the accuracy is
	//!   computed and inserted into the accuracy accumulators
	//!
	//! \param[in] taskStatistics The task's statistics
	inline void accumulateStatistics(TaskStatistics *taskStatistics)
	{
		assert(taskStatistics != nullptr);

		// Normalize the execution time using the task's computational cost
		double cost = (double) taskStatistics->getCost();
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

		// Accumulate the unitary time and the accuracy obtained
		_accumulatorLock.lock();
		_timeAccumulator(normalizedTime);
		if (predictionAvailable) {
			_accuracyAccumulator(accuracy);
		}
		_accumulatorLock.unlock();
	}

};

#endif // TASKTYPE_STATISTICS_HPP
