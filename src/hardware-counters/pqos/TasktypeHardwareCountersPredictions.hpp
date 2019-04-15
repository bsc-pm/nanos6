/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASKTYPE_HARDWARE_COUNTERS_PREDICTIONS_HPP
#define PQOS_TASKTYPE_HARDWARE_COUNTERS_PREDICTIONS_HPP

#include <boost/accumulators/statistics/rolling_mean.hpp>

#include "TaskHardwareCounters.hpp"
#include "TaskHardwareCountersPredictions.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"


#define PREDICTION_UNAVAILABLE -1


class TasktypeHardwareCountersPredictions {

private:
	
	typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::sum, boost::accumulators::tag::mean, boost::accumulators::tag::variance, boost::accumulators::tag::rolling_mean> > counter_rolling_accumulator_t;
	
	//! The rolling window's size for accumulators
	static EnvironmentVariable<int> _rollingWindow;
	
	//! Accumulators for hardware counters
	std::vector<counter_rolling_accumulator_t> _counters;
	
	//! Accumulators for normalized hardware counters
	std::vector<counter_rolling_accumulator_t> _normalizedCounters;

	//! Accumulators for the accuracy of predictions of hardware counters
	std::vector<counter_accumulator_t> _accuracies;
	
	//! Spinlock to ensure atomic access within the accumulators
	SpinLock _accumulatorLock;
	
	//! Number of instances of the tasktype taken into account for accumulators
	size_t _instances;
	
	
public:
	
	inline TasktypeHardwareCountersPredictions() :
		_counters(HWCounters::num_counters, counter_rolling_accumulator_t(boost::accumulators::tag::rolling_window::window_size = _rollingWindow)),
		_normalizedCounters(HWCounters::num_counters, counter_rolling_accumulator_t(boost::accumulators::tag::rolling_window::window_size = _rollingWindow)),
		_accuracies(HWCounters::num_counters),
		_accumulatorLock(),
		_instances(0)
	{
	}
	
	
	inline size_t getInstances() const
	{
		return _instances;
	}
	
	inline double getCounterAverage(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double mean = boost::accumulators::mean(_counters[counterId]);
		_accumulatorLock.unlock();
		return mean;
	}
	
	inline double getCounterRollingAverage(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double rollingAverage = boost::accumulators::rolling_mean(_counters[counterId]);
		_accumulatorLock.unlock();
		return rollingAverage;
	}
	
	inline double getCounterStdev(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double stdev = sqrt(boost::accumulators::variance(_counters[counterId]));
		_accumulatorLock.unlock();
		return stdev;
	}
	
	inline double getCounterSum(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double sum = boost::accumulators::sum(_counters[counterId]);
		_accumulatorLock.unlock();
		return sum;
	}
	
	inline double getNormalizedCounterRollingAverage(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double normalizedRollingAverage = boost::accumulators::rolling_mean(_normalizedCounters[counterId]);
		_accumulatorLock.unlock();
		return normalizedRollingAverage;
	}
	
	inline double getAverageAccuracy(HWCounters::counters_t counterId)
	{
		_accumulatorLock.lock();
		double averageAccuracy = boost::accumulators::mean(_accuracies[counterId]);
		_accumulatorLock.unlock();
		return averageAccuracy;
	}
	
	
	//    ACCUMULATOR MANIPULATION    //
	
	//! \brief Insert unitary (normalized) values for counters. These values
	//! are counter values per unit of cost. Both parameters are vectors which
	//! should contain, respectively, the counter identifiers and their values.
	//! Note: The index for an identifier in the counterId vector, and the
	//! index for a normalized value in the counterValues vector should be the
	//! same if they both refer to the same counter
	//! \param counterId An array with counter identifiers
	//! \param counterValues An array with unitary values
	inline void insertCounterValuesPerUnitOfCost(
		std::vector<HWCounters::counters_t> &counterIds,
		std::vector<double> &counterValues
	) {
		assert(counterIds.size() == counterValues.size());
		
		_accumulatorLock.lock();
		
		for (unsigned short id = 0; id < counterIds.size(); ++id) {
			_normalizedCounters[counterIds[id]](counterValues[id]);
		}
		++_instances;
		
		_accumulatorLock.unlock();
	}
	
	//! \brief Accumulate a task's hardware counters into tasktype predictions
	//! \param taskCounters The task's hardware counters
	//! \param taskCountersPredictions The task's hardware counter predictions
	inline void accumulateCounters(
		TaskHardwareCounters *taskCounters,
		TaskHardwareCountersPredictions *taskPredictions
	) {
		assert(taskCounters != nullptr);
		assert(taskPredictions != nullptr);
		
		double counters[HWCounters::num_counters];
		double normalizedCounters[HWCounters::num_counters];
		double accuracies[HWCounters::num_counters];
		double max, abs;
		
		// Pre-compute all the needed values before entering the lock
		for (unsigned short id = 0; id < HWCounters::num_counters; ++id) {
			counters[id] = taskCounters->getCounter((HWCounters::counters_t) id);
			normalizedCounters[id] = (counters[id] / taskCounters->getCost());
			
			if (taskPredictions->hasPrediction((HWCounters::counters_t) id)) {
				double predicted = taskPredictions->getCounterPrediction((HWCounters::counters_t) id);
				// Avoid divisions by 0
				if (counters[id] == predicted) {
					accuracies[id] = 100.0;
				}
				else {
					max = std::max(std::abs(counters[id]), std::abs(predicted));
					abs = std::abs(counters[id] - predicted);
					accuracies[id] = 100.0 - ((abs / max) * 100.0);
				}
			}
		}
		
		// Aggregate all the information into the accumulators
		_accumulatorLock.lock();
		for (unsigned short id = 0; id < HWCounters::num_counters; ++id) {
			_counters[id](counters[id]);
			_normalizedCounters[id](normalizedCounters[id]);
			if (taskPredictions->hasPrediction((HWCounters::counters_t) id)) {
				_accuracies[id](accuracies[id]);
			}
		}
		// This task is now accounted in the accumulators
		++_instances;
		_accumulatorLock.unlock();
	}
	
	//! \brief Get a hardware counter prediction for a task
	//! \param counterId The hardware counter's id
	//! \param cost The task's computational costs
	inline double getCounterPrediction(HWCounters::counters_t counterId, size_t cost)
	{
		double unitaryValue;
		bool canPredict = false;
		
		// Check if a prediction can be inferred
		_accumulatorLock.lock();
		if (_instances) {
			canPredict = true;
			unitaryValue = boost::accumulators::rolling_mean(_normalizedCounters[counterId]);
		}
		_accumulatorLock.unlock();
		
		if (canPredict) {
			return ((double) cost * unitaryValue);
		}
		else {
			return PREDICTION_UNAVAILABLE;
		}
	}
};

#endif // PQOS_TASKTYPE_HARDWARE_COUNTERS_PREDICTIONS_HPP
