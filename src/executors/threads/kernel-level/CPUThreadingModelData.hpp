/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_THREADING_MODEL_DATA_HPP
#define CPU_THREADING_MODEL_DATA_HPP


#include <atomic>
#include <deque>

#include "lowlevel/EnvironmentVariable.hpp"


struct CPU;
class WorkerThread;


struct CPUThreadingModelData {
private:
	//! \brief a thread responsible for shutting down the rest of the threads and itself
	std::atomic<WorkerThread *> _shutdownControllerThread;
	
	//! \brief number of threads that must be shut down
	static std::atomic<long> _shutdownThreads;
	
	//! \brief last thread that joins any other thread
	static std::atomic<WorkerThread *> _mainShutdownControllerThread;
	
	static EnvironmentVariable<StringifiedMemorySize> _defaultThreadStackSize;
	
	friend class WorkerThreadBase;
	
public:
	CPUThreadingModelData()
		: _shutdownControllerThread(nullptr)
	{
	}
	
	void initialize(CPU *cpu);
	void shutdownPhase1(CPU *cpu);
	void shutdownPhase2(CPU *cpu);
	
	static size_t getDefaultStackSize()
	{
		return (size_t) _defaultThreadStackSize.getValue();
	}
};


#endif // CPU_THREADING_MODEL_DATA_HPP
