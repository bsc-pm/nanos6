#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <bitset>
#include <deque>

#include "DataAccessRange.hpp"


class Task;


struct CPUDependencyData {
	typedef std::deque<Task *> satisfied_originator_list_t;
	typedef std::deque<Task *> removable_task_list_t;
	typedef std::deque<DataAccessRange> data_access_range_list_t;
	
	//! Tasks whose accesses have been satified after ending a task
	satisfied_originator_list_t _satisfiedOriginators;
	removable_task_list_t _removableTasks;
	data_access_range_list_t _removedRangesFromBottomMap;
	
#ifndef NDEBUG
	std::atomic<bool> _inUse;
#endif
};


#endif // CPU_DEPENDENCY_DATA_HPP
