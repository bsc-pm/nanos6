#ifndef THREAD_MANAGER_DEBUGGING_INTERFACE_HPP
#define THREAD_MANAGER_DEBUGGING_INTERFACE_HPP


#include "ThreadManager.hpp"


class ThreadManagerDebuggingInterface {
public:
	typedef std::vector<std::atomic<CPU *>> cpu_list_t;
	
	static cpu_list_t const &getCPUListConstRef()
	{
		return ThreadManager::_cpus;
	}
	
	static cpu_list_t &getCPUListRef()
	{
		return ThreadManager::_cpus;
	}
	
	static cpu_set_t const &getProcessCPUMaskRef()
	{
		return ThreadManager::_processCPUMask;
	}
	
};


#endif // THREAD_MANAGER_DEBUGGING_INTERFACE_HPP
