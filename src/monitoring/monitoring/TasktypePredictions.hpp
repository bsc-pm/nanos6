/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_PREDICTIONS_HPP
#define TASKTYPE_PREDICTIONS_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>

#include "TaskPredictions.hpp"
#include "TaskStatistics.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"


#define PREDICTION_UNAVAILABLE -1


class TasktypePredictions {

private:
	
	typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::rolling_mean, boost::accumulators::tag::variance> > time_accumulator_t;
	
	typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::sum, boost::accumulators::tag::mean> > accumulator_t;
	
	//! Contains the rolling-window size for accumulators
	static EnvironmentVariable<int> _rollingWindow;
	
	//! Number of instances taken into account for predictions of this tasktype
	size_t _instances;
	
	//! An accumulator of unitary timing values of a tasktype
	time_accumulator_t _timeAccumulator;
	
	//! Accuracy of timing predictions for a tasktype
	accumulator_t _accuracyAccumulator;
	
	//! Spinlock to ensure atomic access within the accumulators
	SpinLock _accumulatorLock;
	
	
public:
	
	inline TasktypePredictions() :
		_timeAccumulator(boost::accumulators::tag::rolling_window::window_size = _rollingWindow),
		_accuracyAccumulator(),
		_accumulatorLock()
	{
		_instances = 0;
	}
	
	
	//    SETTERS & GETTERS    //
	
	inline size_t getInstances() const
	{
		return _instances;
	}
	
	//! \brief Get the rolling-average time per unit of cost of this tasktype
	inline double getAverageTimePerUnitOfCost()
	{
		_accumulatorLock.lock();
		double average = boost::accumulators::rolling_mean(_timeAccumulator);
		_accumulatorLock.unlock();
		return average;
	}
	
	//! \brief Get the standard deviation of the time per unit of cost of this
	//! tasktype
	inline double getStdevTimePerUnitOfCost()
	{
		_accumulatorLock.lock();
		double stdev = sqrt(boost::accumulators::variance(_timeAccumulator));
		_accumulatorLock.unlock();
		return stdev;
	}
	
	//! \brief Get the average accuracy of all predictions
	inline double getPredictionAccuracy()
	{
		_accumulatorLock.lock();
		double average = boost::accumulators::mean(_accuracyAccumulator);
		_accumulatorLock.unlock();
		return average;
	}
	
	
	//    PREDICTIONS    //
	
	//! \brief Insert a unitary value of time (time per unit of cost)
	//! in the time accumulators, used later to inferr predictions
	inline void insertTimePerUnitOfCost(double unitaryTime)
	{
		_accumulatorLock.lock();
		_timeAccumulator(unitaryTime);
		++_instances;
		_accumulatorLock.unlock();
	}
	
	//! \brief Accumulate a task's timing prediction statistics into its
	//! tasktype prediction statistics
	//!
	//! \param taskStatistics The task's statistics
	//! \param taskPredictions The task's predictions
	inline void accumulatePredictions(
		const TaskStatistics *taskStatistics,
		const TaskPredictions *taskPredictions
	) {
		assert(taskStatistics != nullptr);
		assert(taskPredictions != nullptr);
		
		// Normalize the execution time using the task's computational cost
		const double cost        = (double) taskStatistics->getCost();
		const double elapsed     = taskStatistics->getElapsedTime();
		const double unitaryTime = elapsed / cost;
		
		double predicted = 0.0;
		double error     = 0.0;
		double accuracy  = 0.0;
		
		// Elapsed time should always be different than 0
		// Otherwise the accuracy is undefined per MSE definition
		bool predictionIsCorrect = (taskPredictions->hasPrediction() && (elapsed > 0));
		if (predictionIsCorrect) {
			predicted = taskPredictions->getTimePrediction();
			error     = 100.0 * (std::abs(predicted - elapsed) / std::max(elapsed, predicted));
			accuracy  = 100.0 - error;
		}
		
		// Accumulate the unitary time and the accuracy obtained
		_accumulatorLock.lock();
		_timeAccumulator(unitaryTime);
		if (predictionIsCorrect) {
			_accuracyAccumulator(accuracy);
		}
		// Increase the number of instances of this tasktype
		++_instances;
		_accumulatorLock.unlock();
	}
	
	//! \brief Get a timing prediction for a task
	//! \param cost The task's computational costs
	inline double getTimePrediction(size_t cost)
	{
		double unitaryTime;
		bool canPredict = false;
		
		// Check if a prediction can be inferred
		_accumulatorLock.lock();
		if (_instances) {
			canPredict = true;
			unitaryTime = boost::accumulators::rolling_mean(_timeAccumulator);
		}
		_accumulatorLock.unlock();
		
		if (canPredict) {
			return ((double) cost * unitaryTime);
		}
		
		return PREDICTION_UNAVAILABLE;
	}
	
};

#endif // TASKTYPE_PREDICTIONS_HPP
