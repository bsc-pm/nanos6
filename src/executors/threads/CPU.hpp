#ifndef EXECUTORS_THREADS_CPU_HPP
#define EXECUTORS_THREADS_CPU_HPP


#include "hardware/places/CPUPlace.hpp"
#include "lowlevel/SpinLock.hpp"

#include <atomic>
#include <deque>

#include <pthread.h>


class WorkerThread;


namespace threaded_executor_internals {

	struct CPU: public CPUPlace {
		//! \brief this lock affects _enabled, _runningThread
		SpinLock _statusLock;
		
		bool _enabled;
		bool _mustExit;
		
		//! \brief the thread that is currently running in this CPU or nullptr
		WorkerThread *_runningThread;
		
		SpinLock _idleThreadsLock;
		
		//! \brief threads currently assigned to this CPU that are currently suspended due to idleness
		std::deque<WorkerThread *> _idleThreads;
		
		SpinLock _readyThreadsLock;
		
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

}


#endif // EXECUTORS_THREADS_CPU_HPP

