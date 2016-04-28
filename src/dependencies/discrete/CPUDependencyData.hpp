#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <deque>


class Task;


struct CPUDependencyData {
	typedef std::deque<Task *> satisfied_originator_list_t;
	
	//! Tasks whose accesses have been satified after ending a task
	satisfied_originator_list_t _satisfiedAccessOriginators;
};


#endif // CPU_DEPENDENCY_DATA_HPP
