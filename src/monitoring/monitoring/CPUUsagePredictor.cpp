/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUUsagePredictor.hpp"
#include "WorkloadPredictor.hpp"


ConfigVariable<size_t> CPUUsagePredictor::_predictionRate("monitoring.cpuusage_prediction_rate", 100 /* µs */);
double CPUUsagePredictor::_prediction;
bool CPUUsagePredictor::_predictionAvailable;
CPUUsagePredictor::accumulator_t CPUUsagePredictor::_accuracies;


double CPUUsagePredictor::getCPUUsagePrediction(size_t time)
{
	// Retrieve the current ready workload
	double readyLoad      = WorkloadPredictor::getPredictedWorkload(ready_load);
	double executingLoad  = WorkloadPredictor::getPredictedWorkload(executing_load);
	double readyTasks     = (double) WorkloadPredictor::getNumInstances(ready_load);
	double executingTasks = (double) WorkloadPredictor::getNumInstances(executing_load);
	double numberCPUs     = (double) CPUMonitor::getNumCPUs();

	// To account accuracy, make sure this isn't the first prediction
	if (!_predictionAvailable) {
		_predictionAvailable = true;
	} else {
		// Compute the accuracy of the last prediction
		double utilization = CPUMonitor::getTotalActiveness();
		double error = (std::abs(utilization - _prediction)) / numberCPUs;
		double accuracy = 100.0 - (100.0 * error);
		_accuracies(accuracy);
	}

	// Make sure that the prediction is:
	// - At most the amount of tasks or the amount of time available
	// - At least 1 CPU for runtime-related operations
	_prediction = std::min(executingTasks + readyTasks, (readyLoad + executingLoad) / (double) time);
	_prediction = std::max(_prediction, 1.0);
	return _prediction;
}
