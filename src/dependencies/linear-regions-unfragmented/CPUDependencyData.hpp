/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
