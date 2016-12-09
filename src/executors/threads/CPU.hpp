#ifndef EXECUTORS_THREADS_CPU_HPP
#define EXECUTORS_THREADS_CPU_HPP


#include "CPUDependencyData.hpp"

#include "hardware/places/CPUPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/SpinLock.hpp"

#include <atomic>
#include <cassert>
#include <deque>

#include <pthread.h>


class WorkerThread;


struct CPU: public CPUPlace {
	
#ifdef __KNC__
	// For some reason ICPC from composer_xe_2015.2.164 with KNC is unable to handle the activation_status_t as is correctly
	typedef unsigned int activation_status_t;
#define activation_status_t knc_workaround_activation_status_t
#endif
	
	typedef enum {
        uninitialized_status=0,
		starting_status,
		enabled_status,
		enabling_status,
		disabling_status,
		disabled_status
	} activation_status_t;
	
#ifdef __KNC__
#undef activation_status_t
#endif
	
	std::atomic<activation_status_t> _activationStatus;
	
	size_t _systemCPUId;
	size_t _virtualCPUId;
	
	//! \brief the CPU mask so that we can later on migrate threads to this CPU
	cpu_set_t _cpuMask;
	
	//! \brief the pthread attr that is used for all the threads of this CPU
	pthread_attr_t _pthreadAttr;
	
	//! \brief a thread responsible for shutting down the rest of the threads and itself
	std::atomic<WorkerThread *> _shutdownControlerThread;
	
	CPUDependencyData _dependencyData;
	
	CPU(size_t systemCPUId, size_t virtualCPUId);
	
	// Not copyable
	CPU(CPU const &) = delete;
	CPU operator=(CPU const &) = delete;
	
	~CPU()
	{
	}
	
	inline void bindThread(pthread_t internalPThread)
	{
		int rc = pthread_setaffinity_np(internalPThread, CPU_ALLOC_SIZE(_systemCPUId+1), &_cpuMask);
		FatalErrorHandler::handle(rc, " when changing affinity of pthread ", internalPThread, " to CPU ", _systemCPUId);
	}

    inline void initializeIfNeeded() 
    {
        activation_status_t expectedStatus = uninitialized_status;
        _activationStatus.compare_exchange_strong(expectedStatus, starting_status);
    }
	
};


#endif // EXECUTORS_THREADS_CPU_HPP

