/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentReductions.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <ReductionInfo.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
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
		logEntry->_contents << " <-> DeallocatedReductionInfo " << reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	void enterAllocateReductionInfo(data_access_id_t dataAccessId, const DataAccessRegion& accessRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> EnterAllocateReductionInfo "
			<< " region: " << accessRegion.getStartAddress() << ":" << accessRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void exitAllocateReductionInfo(data_access_id_t dataAccessId, const ReductionInfo& reductionInfo, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- ExitAllocateReductionInfo " << &reductionInfo
			<< " region: " << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " dataAccess:" << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void enterRetrievePrivateReductionStorage(const DataAccessRegion& originalRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> EnterRetrievePrivateReductionStorage "
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void exitRetrievePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const DataAccessRegion& originalRegion, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- ExitRetrievePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void enterAllocatePrivateReductionStorage(const ReductionInfo& reductionInfo, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> EnterAllocatePrivateReductionStorage "
			<< " reductionInfo:" << &reductionInfo
			<< " region:" << originalRegion.getStartAddress() << ":" << originalRegion.getSize()
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	void exitAllocatePrivateReductionStorage(const ReductionInfo& reductionInfo, const DataAccessRegion& privateStorage, const InstrumentationContext &context) {
		if (!_verboseReductions) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		DataAccessRegion originalRegion = reductionInfo.getOriginalRegion();
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- ExitAllocatePrivateReductionStorage " << privateStorage.getStartAddress() << ":" << privateStorage.getSize()
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
