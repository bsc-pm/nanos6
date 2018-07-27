/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "WorkerThreadBase.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <InstrumentComputePlaceManagement.hpp>


void WorkerThreadBase::shutdownSequence()
{
	CPU *cpu = getComputePlace();
	assert(cpu != nullptr);
	
	CPUThreadingModelData &threadingModelData = cpu->getThreadingModelData();
	
	if (threadingModelData._shutdownControllerThread.load() == this) {
		// This thread is the shutdown controller (of the CPU)
		
		Instrument::shuttingDownComputePlace(cpu->getInstrumentationId());
		
		bool isMainController = (threadingModelData._mainShutdownControllerThread.load() == this);
		
		// Keep processing threads
		bool done = false;
		while (!done) {
			// Find next to wake up
			WorkerThread *next = ThreadManager::getAnyIdleThread();
			
			if (next != nullptr) {
				assert(next->getTask() == nullptr);
				
				next->signalShutdown();
				
				// Resume the thread
				next->resume(cpu, true);
				next->join();
			} else {
				// No more idle threads (for the moment)
				if (!isMainController) {
					// Let the main shutdown controller handle any thread that may be lagging (did not enter the idle queue yet)
					done = true;
				} else if (threadingModelData._shutdownThreads == 1) {
					// This is the main shutdown controller and is also the last (worker) thread
					assert(isMainController);
					done = true;
				}
			}
		}
	}
	
	// The current thread must exit after this call
	threadingModelData._shutdownThreads--;
}
