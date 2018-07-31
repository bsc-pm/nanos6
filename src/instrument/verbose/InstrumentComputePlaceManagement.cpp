/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/


#include "InstrumentComputePlaceManagement.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	compute_place_id_t createdCPU(unsigned int virtualCPUId, size_t NUMANode)
	{
		compute_place_id_t computePlace(virtualCPUId, Verbose::_concurrentUnorderedListSlotManager.getSlot());
		
		if (!Verbose::_verboseComputePlaceManagement) {
			return computePlace;
		}
		
		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();
		
		Verbose::LogEntry *logEntry = Verbose::getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " Created Compute Place " << computePlace << " in NUMA node " << NUMANode;
		
		addLogEntry(logEntry);
		
		return computePlace;
	}
	
	
	compute_place_id_t createdCUDAGPU()
	{
		return compute_place_id_t(-2, Verbose::_concurrentUnorderedListSlotManager.getSlot());
	}
	
	
	void suspendingComputePlace(compute_place_id_t const &computePlace)
	{
		if (!Verbose::_verboseComputePlaceManagement) {
			return;
		}
		
		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();
		
		Verbose::LogEntry *logEntry = Verbose::getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " Suspending Compute Place " << computePlace;
		
		addLogEntry(logEntry);
	}
	
	void resumedComputePlace(compute_place_id_t const &computePlace)
	{
		if (!Verbose::_verboseComputePlaceManagement) {
			return;
		}
		
		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();
		Verbose::LogEntry *logEntry = Verbose::getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " Resuming Compute Place " << computePlace;
		
		addLogEntry(logEntry);
	}
	
	void shuttingDownComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
		if (!Verbose::_verboseComputePlaceManagement) {
			return;
		}
		
		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();
		
		Verbose::LogEntry *logEntry = Verbose::getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " Shutting Down Compute Place " << computePlace;
		
		addLogEntry(logEntry);
	}
}

