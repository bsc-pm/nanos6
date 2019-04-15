/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "WisdomManager.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


EnvironmentVariable<bool> WisdomManager::_enabled("NANOS6_WISDOM_ENABLE", false);
EnvironmentVariable<std::string> WisdomManager::_filePath("NANOS6_WISDOM_PATH", "./.nanos6_monitoring_wisdom.json");
WisdomManager *WisdomManager::_manager;


void WisdomManager::loadWisdom()
{
	if (_enabled) {
		if (Monitoring::isEnabled() || HardwareCounters::isEnabled()) {
			struct stat pathStatus;
			// Check if the file exists
			if ( (stat(_filePath.getValue().c_str(), &pathStatus) == 0) ) {
				// Create a root node and load the json file in the node
				Json::ptree rootNode;
				try {
					Json::read_json(_filePath.getValue(), rootNode);
				} catch (Json::json_parser::json_parser_error readError) {
					FatalErrorHandler::warnIf(true, "JSON error when trying to read the wisdom file. Continuing execution without wisdom loaded.");
					return;
				}
				
				// Try to load the main child node from the root node
				bool nodeFound = rootNode.find("nanos6_wisdom") != rootNode.not_found();
				if (!nodeFound) {
					FatalErrorHandler::warnIf(true, "JSON error when trying to read the wisdom file's content, data may be corrupted. Continuing execution without wisdom loaded.");
					return;
				}
				
				// Create a map to be filled with monitoring statistics
				WisdomMap dataToStore;
				
				// Navigate through the file and extract metrics
				for (auto const &childNode : rootNode.get_child("nanos6_wisdom")) {
					const std::string label = childNode.first;
					std::vector<double> unitaryValues;
					std::vector<std::string> descriptions;
					double metricValue;
					for (auto const &metric : childNode.second) {
						try {
							metricValue = metric.second.get_value<double>();
							descriptions.push_back(metric.first);
							unitaryValues.push_back(metricValue);
						} catch (Json::ptree_error conversionError) {
							FatalErrorHandler::warnIf(true,
								"JSON error when converting event '",
								metric.first,
								"' with value '",
								metric.second.get_value<std::string>(),
								"' for tasktype '",
								label, "'; Skipping."
							);
						}
					}
					dataToStore.emplace(label, UnitaryMetrics(descriptions, unitaryValues));
				}
				
				// Retrieve Monitoring statistics
				if (Monitoring::isEnabled()) {
					for (auto const &it : dataToStore) {
						for (size_t id = 0; id < it.second._descriptions.size(); ++id) {
							if (it.second._descriptions[id] == "unitary_time") {
								TaskMonitor::insertTimePerUnitOfCost(it.first, it.second._unitaryValues[id]);
							}
						}
					}
				}
				
				// Retrieve HardwareCounters statistics
				if (HardwareCounters::isEnabled()) {
					for (auto const &it : dataToStore) {
						std::vector<HWCounters::counters_t> counterIds;
						std::vector<double> counterValues;
						for (size_t id = 0; id < it.second._descriptions.size(); ++id) {
							HWCounters::counters_t counterIndex = HWCounters::invalid_counter;
							for (unsigned short i = 0; i < HWCounters::num_counters; ++i) {
								if (HWCounters::counterDescriptions[i] == it.second._descriptions[id]) {
									counterIndex = (HWCounters::counters_t) i;
									break;
								}
							}
							if (counterIndex != HWCounters::invalid_counter) {
								counterIds.push_back(counterIndex);
								counterValues.push_back(it.second._unitaryValues[id]);
							}
						}
						TaskHardwareCountersMonitor::insertCounterValuesPerUnitOfCost(it.first, counterIds, counterValues);
					}
				}
			}
		}
	}
}


void WisdomManager::storeWisdom()
{
	if (_enabled) {
		if (Monitoring::isEnabled() || HardwareCounters::isEnabled()) {
			// Create/open a wisdom file
			std::ofstream wisdomFile (_filePath.getValue().c_str());
			FatalErrorHandler::failIf(!wisdomFile.is_open(), "Unable to create a wisdom file under path ", _filePath.getValue());
			
			// Create a map to be filled with monitoring statistics
			WisdomMap dataToStore;
			
			// Retrieve Monitoring statistics
			if (Monitoring::isEnabled()) {
				std::vector<std::string> labels;
				std::vector<double> unitaryTimes;
				TaskMonitor::getAverageTimesPerUnitOfCost(labels, unitaryTimes);
				for (size_t i = 0; i < labels.size(); ++i) {
					if (labels[i] != "main") {
						dataToStore[labels[i]]._descriptions.push_back("unitary_time");
						dataToStore[labels[i]]._unitaryValues.push_back(unitaryTimes[i]);
					}
				}
			}
			
			// Retrieve HardwareCounters statistics
			if (HardwareCounters::isEnabled()) {
				std::vector<std::string> labels;
				std::vector<std::vector<std::pair<HWCounters::counters_t, double>>> unitaryValues;
				TaskHardwareCountersMonitor::getAverageCounterValuesPerUnitOfCost(labels, unitaryValues);
				for (size_t i = 0; i < labels.size(); ++i) {
					if (labels[i] != "main") {
						for (size_t id = 0; id < unitaryValues[i].size(); ++id) {
							const std::string label = HWCounters::counterDescriptions[unitaryValues[i][id].first];
							dataToStore[labels[i]]._descriptions.push_back(label);
							dataToStore[labels[i]]._unitaryValues.push_back(unitaryValues[i][id].second);
						}
					}
				}
			}
			
			// Create a root node, a node for tasktypes and a node for tasktype unitary values
			Json::ptree rootNode;
			Json::ptree taskTypeNode;
			Json::ptree taskTypeValuesNode;
			
			// Parse monitoring data into the root node
			for (auto const &it : dataToStore) {
				// Store each of the unitary values
				for (size_t i = 0; i < it.second._descriptions.size(); ++i) {
					taskTypeValuesNode.put(it.second._descriptions[i], it.second._unitaryValues[i]);
				}
				// Use push_back as the label may contain special chars
				taskTypeNode.push_back(Json::ptree::value_type(it.first, taskTypeValuesNode));
			}
			rootNode.add_child("nanos6_wisdom", taskTypeNode);
			
			// Write the wisdom into the file
			Json::write_json(_filePath.getValue(), rootNode);
			wisdomFile.flush();
			wisdomFile.close();
		}
	}
}
