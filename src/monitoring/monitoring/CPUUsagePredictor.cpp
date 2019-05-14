/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "CPUUsagePredictor.hpp"
#include "WorkloadPredictor.hpp"


CPUUsagePredictor *CPUUsagePredictor::_predictor;
EnvironmentVariable<size_t> CPUUsagePredictor::_predictionRate("NANOS6_CPU_USAGE_PREDICTION_RATE", 100);


double CPUUsagePredictor::getPredictedCPUUsage(size_t time)
{
	assert(_predictor != nullptr);
	
	// Retrieve the current ready workload
	double readyLoad      = WorkloadPredictor::getPredictedWorkload(ready_load);
	double executingLoad  = WorkloadPredictor::getPredictedWorkload(executing_load);
	double readyTasks     = (double) WorkloadPredictor::getNumInstances(ready_load);
	double executingTasks = (double) WorkloadPredictor::getNumInstances(executing_load);
	double numberCPUs     = (double) CPUMonitor::getNumCPUs();
	
	double prediction = _predictor->_prediction;
	
	// To account accuracy, make sure this isn't the first prediction
	if (_predictor->_predictionAvailable) {
		// Compute the accuracy of the last prediction
		double utilization = CPUMonitor::getTotalActiveness();
		double error = (std::abs(utilization - prediction)) / numberCPUs;
		double accuracy = 100.0 - (100.0 * error);
		_predictor->_accuracies(accuracy);
	}
	
	// Make sure that the prediction is:
	// -) at most the amount of tasks or the amount of time available
	// -) at most 'numberCPUs' at 100% usage
	// -) at least 1 CPU for runtime-related operations
	prediction = std::min(executingTasks + readyTasks, (readyLoad + executingLoad) / (double) time);
	prediction = std::min(prediction, numberCPUs);
	prediction = std::max(prediction, (double) 1.0);
	
	return prediction;
}
