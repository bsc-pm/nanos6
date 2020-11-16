/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskStatistics.hpp"
#include "TasktypeStatistics.hpp"
#include "hardware-counters/HardwareCounters.hpp"

ConfigVariable<int> TasktypeStatistics::_rollingWindow("monitoring.rolling_window", 20);


double TasktypeStatistics::getTimingPrediction(size_t cost)
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

double TasktypeStatistics::getCounterPrediction(size_t counterId, size_t cost)
{
	double normalizedValue = PREDICTION_UNAVAILABLE;

	// Check if a prediction can be inferred
	_counterAccumulatorsLock.lock();
	if (BoostAcc::count(_normalizedCounterAccumulators[counterId])) {
		normalizedValue = ((double) cost) * BoostAcc::rolling_mean(_normalizedCounterAccumulators[counterId]);
	}
	_counterAccumulatorsLock.unlock();

	return normalizedValue;
}

void TasktypeStatistics::accumulateStatisticsAndCounters(
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
	bool predictionAvailable = (taskStatistics->hasTimePrediction() && (elapsed > 0.0));
	if (predictionAvailable) {
		double predicted = taskStatistics->getTimePrediction();
		double error = 100.0 * (std::abs(predicted - elapsed) / std::max(elapsed, predicted));
		accuracy = 100.0 - error;
	}

	// Accumulate the unitary time, the elapsed time to compute effective
	// parallelism metrics, and the accuracy obtained of a previous prediction
	_timingAccumulatorLock.lock();
	_timingAccumulator(normalizedTime);
	_accumulatedTimeAccumulator(elapsed);
	if (predictionAvailable) {
		_timingAccuracyAccumulator(accuracy);
	}
	_timingAccumulatorLock.unlock();

	//    HARDWARE COUNTERS    //

	const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();
	size_t numEnabledCounters = enabledCounters.size();

	// Pre-compute all the needed values before entering the lock
	// NOTE: We use VLAs even though they are not C++ compliant and could be dangerous,
	// however, the number of enabled counters should not be too high
	double counters[numEnabledCounters];
	double normalizedCounters[numEnabledCounters];
	bool counterPredictionsAvailable[numEnabledCounters];
	double counterAccuracies[numEnabledCounters];
	for (size_t id = 0; id < numEnabledCounters; ++id) {
		counters[id] = taskCounters.getAccumulated(enabledCounters[id]);
		normalizedCounters[id] = (counters[id] / (double) cost);

		counterPredictionsAvailable[id] =
			(taskStatistics->hasCounterPrediction(id) && counters[id] > 0.0);
		if (counterPredictionsAvailable[id]) {
			double predicted = taskStatistics->getCounterPrediction(id);
			double error = 100.0 * (std::abs(predicted - counters[id]) / std::max(counters[id], predicted));
			counterAccuracies[id] = 100.0 - error;
		}
	}

	// Aggregate all the information into the accumulators
	_counterAccumulatorsLock.lock();
	for (size_t id = 0; id < numEnabledCounters; ++id) {
		_counterAccumulators[id](counters[id]);
		_normalizedCounterAccumulators[id](normalizedCounters[id]);

		if (counterPredictionsAvailable[id]) {
			_counterAccuracyAccumulators[id](counterAccuracies[id]);
		}
	}
	_counterAccumulatorsLock.unlock();
}
