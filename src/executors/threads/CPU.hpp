#ifndef EXECUTORS_THREADS_CPU_HPP
#define EXECUTORS_THREADS_CPU_HPP


#include "hardware/places/CPUPlace.hpp"
#include "lowlevel/SpinLock.hpp"

#include <atomic>
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
		starting_status=0,
		enabled_status,
		enabling_status,
		disabling_status,
		disabled_status
	} activation_status_t;
	
#ifdef __KNC__
#undef activation_status_t
#endif
	
	//! \brief this lock affects _runningThread, _idleThreads and _readyThreads
	SpinLock _statusLock;
	
	std::atomic<activation_status_t> _activationStatus;
	
	//! \brief the thread that is currently running in this CPU or nullptr
	WorkerThread *_runningThread;
	
	//! \brief threads currently assigned to this CPU that are currently suspended due to idleness
	std::deque<WorkerThread *> _idleThreads;
	
	//! \brief threads assigned to this CPU that have been blocked, but that are now ready to be resumed
	std::deque<WorkerThread *> _readyThreads;
	
	size_t _systemCPUId;
	
	//! \brief the CPU mask so that we can later on migrate threads to this CPU
	cpu_set_t _cpuMask;
	
	//! \brief the pthread attr that is used for all the threads of this CPU
	pthread_attr_t _pthreadAttr;
	
	CPU(size_t systemCPUId);
	
	// Not copyable
	CPU(CPU const &) = delete;
	CPU operator=(CPU const &) = delete;
	
	~CPU()
	{
	}
	
	void bindThread(pthread_t *internalPThread);
	
};


#endif // EXECUTORS_THREADS_CPU_HPP

