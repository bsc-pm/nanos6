/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MONITORING_HPP
#define MONITORING_HPP

#include "CPUMonitor.hpp"
#include "CPUUsagePredictor.hpp"
#include "TaskMonitor.hpp"
#include "WorkloadPredictor.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "support/JsonFile.hpp"
#include "tasks/Task.hpp"


class Monitoring {

private:
	
	//! Whether monitoring has to be performed or not
	static EnvironmentVariable<bool> _enabled;
	
	//! Whether verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;
	
	//! Whether the wisdom mechanism is enabled
	static EnvironmentVariable<bool> _wisdomEnabled;
	
	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _outputFile;
	
	//! The "monitor", singleton instance
	static Monitoring *_monitor;
	
	//! A Json file for monitoring data
	JsonFile *_wisdom;
	
	
private:
	
	inline Monitoring() :
		_wisdom(nullptr)
	{
	}
	
	//! \brief Try to load previous monitoring data into accumulators
	inline void loadMonitoringWisdom()
	{
		// Create a representation of the system file as a JsonFile
		_wisdom = new JsonFile("./.nanos6_monitoring_wisdom.json");
		assert(_wisdom != nullptr);
		
		// Try to populate the JsonFile with the system file's data
		_wisdom->loadData();
		
		// Navigate through the file and extract the unitary time of each tasktype
		_wisdom->getRootNode()->traverseChildrenNodes(
			[&](const std::string &label, const JsonNode<> &metricsNode) {
				// For each tasktype, check if the unitary time is available
				if (metricsNode.dataExists("unitary_time")) {
					// Insert the metric data for this tasktype into accumulators
					bool converted = false;
					double metricValue = metricsNode.getData("unitary_time", converted);
					if (converted) {
						TaskMonitor::insertTimePerUnitOfCost(label, metricValue);
					}
				}
			}
		);
	}
	
	//! \brief Store monitoring data for future executions as warmup data
	inline void storeMonitoringWisdom()
	{
		// Gather monitoring data for all tasktypes
		std::vector<std::string> labels;
		std::vector<double> unitaryTimes;
		TaskMonitor::getAverageTimesPerUnitOfCost(labels, unitaryTimes);
		
		assert(_wisdom != nullptr);
		
		// The file's root node
		JsonNode<> *rootNode = _wisdom->getRootNode();
		for (size_t i = 0; i < labels.size(); ++i) {
			// Avoid storing information about the main task
			if (labels[i] != "main") {
				// A node for metrics (currently only unitary time)
				JsonNode<double> taskTypeValuesNode;
				taskTypeValuesNode.addData("unitary_time", unitaryTimes[i]);
				
				// Add the metrics to the root node of the file
				rootNode->addChildNode(labels[i], taskTypeValuesNode);
			}
		}
		
		// Store the data from the JsonFile in the system file
		_wisdom->storeData();
		
		// Delete the file as it is no longer needed
		delete _wisdom;
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	Monitoring(Monitoring const&) = delete;            // Copy construct
	Monitoring(Monitoring&&) = delete;                 // Move construct
	Monitoring& operator=(Monitoring const&) = delete; // Copy assign
	Monitoring& operator=(Monitoring &&) = delete;     // Move assign
	
	
	//    MONITORING    //
	
	//! \brief Initialize monitoring
	static void initialize();
	
	//! \brief Shutdown monitoring
	static void shutdown();
	
	//! \brief Display monitoring statistics
	static void displayStatistics();
	
	//! \brief Whether monitoring is enabled
	static bool isEnabled();
	
	
	//    TASKS    //
	
	//! \brief Gather basic information about a task when it is created
	//! \param task The task to gather information about
	static void taskCreated(Task *task);
	
	//! \brief Propagate monitoring operations after a task has changed its
	//! execution status
	//! \param task The task that's changing status
	//! \param newStatus The new execution status of the task
	//! \param cpu The cpu onto which a thread is running the task
	static void taskChangedStatus(Task *task, monitoring_task_status_t newStatus, ComputePlace *cpu = nullptr);
	
	//! \brief Propagate monitoring operations after a task has
	//! completed user code execution
	//! \param task The task that has completed the execution
	//! \param cpu The cpu in which a thread was running the task
	static void taskCompletedUserCode(Task *task, ComputePlace *cpu);
	
	//! \brief Propagate monitoring operations after a task has finished
	//! \param task The task that has finished
	static void taskFinished(Task *task);
	
	
	//    THREADS    //
	
	//! \brief Propagate monitoring operations when a thread is initialized
	static void initializeThread();
	
	//! \brief Propagate monitoring operations when a thread is shutdown
	static void shutdownThread();
	
	
	//    PREDICTORS    //
	
	//! \brief Poll the expected time until completion of the current execution
	//! \return An estimation of the time to completion in microseconds
	static double getPredictedElapsedTime();
	
};

#endif // MONITORING_HPP
