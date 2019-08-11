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
	
	_mustExit = true;
	const int MIN_SPINS = 100;
	const int MAX_SPINS = 100*1000*1000;
	
	int spins = MIN_SPINS;
	
	bool canJoin = false;
	while (!canJoin) {
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
}
