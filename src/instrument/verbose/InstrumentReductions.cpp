/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentReductions.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

#include <ReductionInfo.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void allocatedReductionInfo(data_access_id_t dataAccessId, const ReductionInfo& reductionInfo, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> AllocatedReductionInfo " << &reductionInfo
			<< " region: " << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void receivedCompatibleReductionInfo(data_access_id_t dataAccessId, const ReductionInfo& reductionInfo, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> ReceivedCompatibleReductionInfo " << &reductionInfo
			<< " region: " << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void deallocatedReductionInfo(data_access_id_t dataAccessId, const ReductionInfo *reductionInfo, const DataAccessRegion& originalRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- DeallocatedReductionInfo " << reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	void retrievedPrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> RetrievedPrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	void enterInitializePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> EnterInitializePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void exitInitializePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- ExitInitializePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void enterCombinePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const DataAccessRegion& originalRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> EnterCombinePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void exitCombinePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const DataAccessRegion& originalRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- ExitCombinePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
}
