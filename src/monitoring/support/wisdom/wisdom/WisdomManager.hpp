/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef WISDOM_MANAGER_HPP
#define WISDOM_MANAGER_HPP

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <fstream>
#include <iostream>
#include <sys/stat.h>

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <HardwareCounters.hpp>
#include <Monitoring.hpp>
#include <TaskHardwareCountersMonitor.hpp>
#include <TaskMonitor.hpp>


namespace Json = boost::property_tree;


struct UnitaryMetrics {
	
	//! The average unitary values of a tasktype for all its metrics
	std::vector<double> _unitaryValues;
	
	//! The descriptions of each unitary metric
	std::vector<std::string> _descriptions;
	
	
	inline UnitaryMetrics() :
		_unitaryValues(),
		_descriptions()
	{
	}
	
	inline UnitaryMetrics(const std::vector<std::string> &descriptions, const std::vector<double> &unitaryValues) :
		_unitaryValues(unitaryValues),
		_descriptions(descriptions)
	{
	}
};


class WisdomManager {

private:
	
	//! Maps a tasktype with its unitary metrics
	typedef std::map< std::string, UnitaryMetrics > WisdomMap;
	
	//! Whether wisdom is enabled
	static EnvironmentVariable<bool> _enabled;
	
	//! The path in which the wisdom file will be searched
	static EnvironmentVariable<std::string> _filePath;
	
	//! The singleton instance
	static WisdomManager *_manager;
	
	
private:
	
	inline WisdomManager()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	WisdomManager(WisdomManager const&) = delete;            // Copy construct
	WisdomManager(WisdomManager&&) = delete;                 // Move construct
	WisdomManager& operator=(WisdomManager const&) = delete; // Copy assign
	WisdomManager& operator=(WisdomManager &&) = delete;     // Move assign
	
	
	//! \brief Initialization of the wisdom manager
	static inline void initialize()
	{
		if (_enabled) {
			// Create the monitoring module
			if (_manager == nullptr) {
				_manager = new WisdomManager();
			}
			
			// Try to load metrics from previous executions
			loadWisdom();
		}
	}
	
	//! \brief Shutdown of the wisdom manager
	static inline void shutdown()
	{
		if (_enabled) {
			// Store metrics for future executions
			storeWisdom();
			
			if (_manager != nullptr) {
				delete _manager;
			}
		}
	}
	
	//! \return Whether the Wisdom is enabled
	static inline bool isEnabled()
	{
		return _enabled;
	}
	
	//! \brief Try to load monitoring data from previous executions
	static void loadWisdom();
	
	//! \brief Stores unitary values for future usage. This function parses
	//! statistics into JSON format & saves them in a file
	static void storeWisdom();
	
};

#endif // WISDOM_MANAGER_HPP
