/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include "TaskHardwareCountersMonitor.hpp"
#include "ThreadHardwareCountersMonitor.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "support/JsonFile.hpp"
#include "tasks/Task.hpp"


class HardwareCounters {

private:

	//! Whether hardware counter monitoring is enabled
	static EnvironmentVariable<bool> _enabled;

	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;

	//! Whether the wisdom mechanism is enabled
	static EnvironmentVariable<bool> _wisdomEnabled;

	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _outputFile;

	//! The singleton instance
	static HardwareCounters *_monitor;

	//! A Json file for hardware counter monitoring data
	JsonFile *_wisdom;


private:

	inline HardwareCounters() :
		_wisdom(nullptr)
	{
	}

	//! \brief Try to load previous hardware counter data into accumulators
	inline void loadHardwareCounterWisdom()
	{
		// Create a representation of the system file as a JsonFile
		_wisdom = new JsonFile("./.nanos6_monitoring_wisdom.json");
		assert(_wisdom != nullptr);

		// Try to populate the JsonFile with the system file's data
		_wisdom->loadData();

		// Navigate through the file and extract metrics for each tasktype
		_wisdom->getRootNode()->traverseChildrenNodes(
			[&](const std::string &label, const JsonNode<> &metricsNode) {
				// Vectors for counter values and indexes (types)
				std::vector<double> values;
				std::vector<HWCounters::counters_t> indexes;

				// For each tasktype and counter check the availability of data
				for (unsigned short i = 0; i < HWCounters::num_counters; ++i) {
					std::string metricLabel(HWCounters::counterDescriptions[i]);
					if (metricsNode.dataExists(metricLabel)) {
						// Retreive and convert the metric
						bool converted = false;
						double metricValue = metricsNode.getData(metricLabel, converted);
						if (converted) {
							indexes.push_back((HWCounters::counters_t) i);
							values.push_back(metricValue);
						}
					}
				}

				// Insert the metric data for this tasktype into accumulators
				TaskHardwareCountersMonitor::insertCounterValuesPerUnitOfCost(label, indexes, values);
			}
		);
	}

	//! \brief Store hardware counter data for future executions as warmup data
	inline void storeHardwareCounterWisdom()
	{
		// NOTE: Data from Monitoring is first saved for the same task types.
		// In order to use the same file in Hardware Counter Monitoring,
		// instead of creating new nodes, existing ones have to be updated.
		// This is done by:
		// - 1) Loading the file data into the current wisdom file (loadData)
		// - 2) Creating a copy of the root node to obtain the data
		// - 3) Clearing the file
		// - 4) Re-writing on the file

		assert(_wisdom != nullptr);

		// 1) Load data stored by Monitoring
		_wisdom->loadData();

		// 2) Make a copy of the root node
		JsonNode<> rootNode(*(_wisdom->getRootNode()));

		// 3) Clear the file
		_wisdom->clearFile();

		// Gather hardware counter data for all tasktypes
		std::vector<std::string> labels;
		std::vector<std::vector<std::pair<HWCounters::counters_t, double>>> unitaryValues;
		TaskHardwareCountersMonitor::getAverageCounterValuesPerUnitOfCost(labels, unitaryValues);

		// The file's real root node
		JsonNode<> *realRootNode = _wisdom->getRootNode();
		for (size_t i = 0; i < labels.size(); ++i) {
			// Avoid storing information about the main task
			if (labels[i] != "main") {
				// A node for all the metrics of this tasktype
				JsonNode<double> taskTypeValuesNode;

				// Try to fill the node with a copy of its previous data if
				// the node already exists. Otherwise, it remains an empty node
				taskTypeValuesNode = JsonNode<double>(rootNode.getChildNode(labels[i]));

				// Retreive all Hardware Counter metrics for this tasktype
				for (size_t id = 0; id < unitaryValues[i].size(); ++id) {
					// Update the existing node with Hardware Counter data
					const std::string &label = HWCounters::counterDescriptions[unitaryValues[i][id].first];
					taskTypeValuesNode.addData(label, unitaryValues[i][id].second);
				}

				// Add the tasktype metrics to the file's root node
				realRootNode->addChildNode(labels[i], taskTypeValuesNode);
			}
		}

		// 4) Store the data from the JsonFile in the system file
		_wisdom->storeData();

		// Delete the file as it is no longer needed
		delete _wisdom;
	}


public:

	// Delete copy and move constructors/assign operators
	HardwareCounters(HardwareCounters const&) = delete;            // Copy construct
	HardwareCounters(HardwareCounters&&) = delete;                 // Move construct
	HardwareCounters& operator=(HardwareCounters const&) = delete; // Copy assign
	HardwareCounters& operator=(HardwareCounters &&) = delete;     // Move assign


	//    HARDWARE COUNTERS    //

	//! \brief Initialization of the hardware counter monitoring module
	static void initialize();

	//! \brief Shutdown of the hardware counter monitoring module
	static void shutdown();

	//! \brief Display hardware counter statistics
	static void displayStatistics();

	//! \brief Whether monitoring is enabled
	static bool isEnabled();


	//    TASKS    //

	//! \brief Gather basic information about a task when it is created
	//! \param[in] task The task to gather information about
	static void taskCreated(Task *task);

	//! \brief Start/resume hardware counter monitoring for a task
	//! \param[in] task The task to start monitoring for
	static void startTaskMonitoring(Task *task);

	//! \brief Stop/pause hardware counter monitoring for a task and aggregate
	//! the current thread's counter into the task's counters
	//! \param[in] task The task to start monitoring for
	static void stopTaskMonitoring(Task *task);

	//! \brief Finish hardware counter monitoring for a task
	//! \param[in] task The task that has finished
	static void taskFinished(Task *task);


	//    THREADS    //

	//! \brief Initialize hardware counter monitoring for the current thread
	static void initializeThread();

	//! \brief Shutdown hardware counter monitoring for the current thread
	static void shutdownThread();

};

#endif // PQOS_HARDWARE_COUNTERS_HPP
