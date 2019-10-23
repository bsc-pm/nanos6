/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>
#include <list>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "CPUManager.hpp"
#include "ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"


ThreadManager::IdleThreads *ThreadManager::_idleThreads;
std::atomic<long> ThreadManager::_totalThreads(0);
ThreadManager::ShutdownThreads *ThreadManager::_shutdownThreads;


void ThreadManager::initialize()
{
	size_t numaNodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_idleThreads = new IdleThreads[numaNodeCount];
	_shutdownThreads = new ShutdownThreads();
}


void ThreadManager::shutdownPhase1()
{
	assert(_shutdownThreads != nullptr);

	// Spin until all threads are marked as shutdown
	const int MIN_SPINS = 100;
	const int MAX_SPINS = 1000*1000;
	int spins = MIN_SPINS;
	bool canJoin = false;
	WorkerThread *idleThread = nullptr;
	while (!canJoin) {
		// Wake up as many threads as possible so that they can participate
		// in the shutdown process
		idleThread = getAnyIdleThread();
		while (idleThread != nullptr) {
			CPU *idleCPU = CPUManager::getIdleCPU();
			if (idleCPU != nullptr) {
				idleThread->resume(idleCPU, true);
			} else {
				// No CPUs available, readd the thread as idle and break
				addIdler(idleThread);
				break;
			}
			idleThread = getAnyIdleThread();
		}

		// Check whether all the threads already added themselves to _shutdownThreads.
		_shutdownThreads->_lock.lock();
		canJoin = (_shutdownThreads->_threads.size() == (size_t) _totalThreads);
		_shutdownThreads->_lock.unlock();

		// Spin for a while to let threads add them to _shutdownThreads.
		int i = 0;
		while (i < spins && !canJoin) {
			i++;
		}

		// Backoff
		if (spins < MAX_SPINS) {
			spins *= 2;
		}
	}

	assert(_shutdownThreads->_threads.size() == (size_t) _totalThreads);

	for (WorkerThread *thread : _shutdownThreads->_threads) {
		thread->join();
	}
}

void ThreadManager::shutdownPhase2()
{
	assert(_shutdownThreads != nullptr);

	for (WorkerThread *thread : _shutdownThreads->_threads) {
		delete thread;
	}
	delete _shutdownThreads;

	delete [] _idleThreads;
}

void ThreadManager::addShutdownThread(WorkerThread *shutdownThread)
{
	assert(shutdownThread != nullptr);
	assert(_shutdownThreads != nullptr);

	CPU *cpu = shutdownThread->getComputePlace();

	_shutdownThreads->_lock.lock();
	_shutdownThreads->_threads.push_back(shutdownThread);
	_shutdownThreads->_lock.unlock();

	// Mark that the CPU is available for anyone else who might need it
	__attribute__((unused)) bool idle = CPUManager::cpuBecomesIdle(cpu, true);
	assert(idle);
}

