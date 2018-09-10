/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>
#include <list>

#include <pthread.h>
#include <unistd.h>

#include <sys/syscall.h>

#include "CPUActivation.hpp"
#include "CPUManager.hpp"
#include "ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include <hardware/HardwareInfo.hpp>


std::atomic<bool> ThreadManager::_mustExit(false);
ThreadManager::IdleThreads *ThreadManager::_idleThreads;
std::atomic<long> ThreadManager::_totalThreads(0);


void ThreadManager::initialize()
{
	size_t numaNodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_idleThreads = new IdleThreads[numaNodeCount];
}


void ThreadManager::shutdown()
{
	_mustExit = true;
	
	// Attempt to wake up all (enabled) CPUs so that they start shutting down the threads
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	std::deque<CPU *> participatingCPUs;
	for (CPU *cpu : cpus) {
		// Sanity check
		if ((cpu != nullptr) && CPUActivation::acceptsWork(cpu)) {
			cpu->getThreadingModelData().shutdownPhase1(cpu);
			
			// Place them in reverse order so the last one we get afterwards is the main shutdown controller
			participatingCPUs.push_front(cpu);
		}
	}
	
	// At this point we have woken as many threads as active CPUs. They perform the
	// shutdown collectively. The number can actually be smaller than activeCPUs.size().
	// The reason is that as soon as one starts the shutdown procedure, it will start
	// collecting other threads. That is, it will be competing to get idle threads too.
	// However, there will be at least one of them, the main shutdown controller, and it
	// will be the last controller in "activeCPUs".
	
	// Join all the shutdown controller threads
	for (auto cpu : participatingCPUs) {
		cpu->getThreadingModelData().shutdownPhase2(cpu);
	}
}


