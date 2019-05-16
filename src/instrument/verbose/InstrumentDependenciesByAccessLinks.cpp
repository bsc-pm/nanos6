/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <atomic>
#include <cassert>

#include "InstrumentDependenciesByAccessLinks.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	static std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId(1);
	
	data_access_id_t createdDataAccess(
		data_access_id_t *superAccessId,
		DataAccessType accessType, bool weak, DataAccessRegion region,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		access_object_type_t objectType,
		task_id_t originatorTaskId,
		InstrumentationContext const &context
	) {
		data_access_id_t id = _nextDataAccessId++;
		
		if (_verboseDependenciesByAccessLinks) {
			LogEntry *logEntry = getLogEntry(context);
			assert(logEntry != nullptr);
			
			logEntry->appendLocation(context);
			logEntry->_contents << " <-> ";
			switch (objectType) {
				case regular_access_type:
					logEntry->_contents << "CreateDataAccess";
					break;
				case entry_fragment_type:
					logEntry->_contents << "CreatedDataSubaccessFragment";
					break;
				case taskwait_type:
					logEntry->_contents << "CreatedTaskwaitFragment";
					break;
				case top_level_sink_type:
					logEntry->_contents << "CreatedTopLevelSink";
					break;
			}
			logEntry->_contents << " " << id;
			if (superAccessId != nullptr) {
				logEntry->_contents << " superaccess:" << *superAccessId << " ";
			}
			
			if (weak) {
				logEntry->_contents << "weak";
			}
			switch (accessType) {
				case READ_ACCESS_TYPE:
					logEntry->_contents << " input";
					break;
				case READWRITE_ACCESS_TYPE:
					logEntry->_contents << " inout";
					break;
				case WRITE_ACCESS_TYPE:
					logEntry->_contents << " output";
					break;
				case CONCURRENT_ACCESS_TYPE:
					logEntry->_contents << " concurrent";
					break;
				case COMMUTATIVE_ACCESS_TYPE:
					logEntry->_contents << " commutative";
					break;
				case REDUCTION_ACCESS_TYPE:
					logEntry->_contents << " reduction";
					break;
				case NO_ACCESS_TYPE:
					logEntry->_contents << " local";
					break;
				default:
					logEntry->_contents << " unknown_access_type";
					break;
			}
			
			logEntry->_contents << " " << region;
			
			if (readSatisfied) {
				logEntry->_contents << " read_safistied";
			}
			if (writeSatisfied) {
				logEntry->_contents << " write_safistied";
			}
			if (globallySatisfied) {
				logEntry->_contents << " safistied";
			}
			if (!readSatisfied && !writeSatisfied && !globallySatisfied) {
				logEntry->_contents << " unsatisfied";
			}
			
			logEntry->_contents << " originator:" << originatorTaskId;
			
			addLogEntry(logEntry);
		}
		
		return id;
	}
	
	
	void upgradedDataAccess(
		data_access_id_t &dataAccessId,
		DataAccessType previousAccessType,
		bool previousWeakness,
		DataAccessType newAccessType,
		bool newWeakness,
		bool becomesUnsatisfied,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> UpgradeDataAccess " << dataAccessId;
		
		logEntry->_contents << " ";
		if (previousWeakness) {
			logEntry->_contents << "weak";
		} else {
			logEntry->_contents << "noweak";
		}
		logEntry->_contents << "->";
		if (newWeakness) {
			logEntry->_contents << "weak";
		} else {
			logEntry->_contents << "noweak";
		}
		
		logEntry->_contents << " ";
		switch (previousAccessType) {
			case READ_ACCESS_TYPE:
				logEntry->_contents << "input";
				break;
			case READWRITE_ACCESS_TYPE:
				logEntry->_contents << "inout";
				break;
			case WRITE_ACCESS_TYPE:
				logEntry->_contents << "output";
				break;
			default:
				logEntry->_contents << "unknown_access_type";
				break;
		}
		logEntry->_contents << "->";
		switch (newAccessType) {
			case READ_ACCESS_TYPE:
				logEntry->_contents << "input";
				break;
			case READWRITE_ACCESS_TYPE:
				logEntry->_contents << "inout";
				break;
			case WRITE_ACCESS_TYPE:
				logEntry->_contents << "output";
				break;
			default:
				logEntry->_contents << "unknown_access_type";
				break;
		}
		
		if (becomesUnsatisfied) {
			logEntry->_contents << " satisfied->unsatisfied";
		}
		
		logEntry->_contents << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void dataAccessBecomesSatisfied(
		data_access_id_t &dataAccessId,
		bool globallySatisfied,
		task_id_t targetTaskId,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> DataAccessBecomesSatisfied " << dataAccessId << " triggererTask:" << context._taskId << " targetTask:" << targetTaskId;
		
		if (globallySatisfied) {
			logEntry->_contents << " +safistied";
		}
		if (!globallySatisfied) {
			logEntry->_contents << " remains_unsatisfied";
		}
		
		addLogEntry(logEntry);
	}
	
	
	void modifiedDataAccessRegion(
		data_access_id_t &dataAccessId,
		DataAccessRegion newRegion,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> ModifiedDataAccessRegion " << dataAccessId << " newRegion:" << newRegion << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	data_access_id_t fragmentedDataAccess(
		data_access_id_t &dataAccessId,
		DataAccessRegion newRegion,
		InstrumentationContext const &context
	) {
		data_access_id_t id = _nextDataAccessId++;
		
		if (_verboseDependenciesByAccessLinks) {
			LogEntry *logEntry = getLogEntry(context);
			assert(logEntry != nullptr);
			
			logEntry->appendLocation(context);
			logEntry->_contents << " <-> FragmentedDataAccess " << dataAccessId << " newFragment:" << id << " newRegion:" << newRegion << " triggererTask:" << context._taskId;
			
			addLogEntry(logEntry);
		}
		
		return id;
	}
	
	
	data_access_id_t createdDataSubaccessFragment(
		data_access_id_t &dataAccessId,
		InstrumentationContext const &context
	) {
		data_access_id_t id = _nextDataAccessId++;
		
		if (_verboseDependenciesByAccessLinks) {
			LogEntry *logEntry = getLogEntry(context);
			assert(logEntry != nullptr);
			
			logEntry->appendLocation(context);
			logEntry->_contents << " <-> CreatedDataSubaccessFragment " << dataAccessId << " newSubaccessFragment:" << id << " triggererTask:" << context._taskId;
			
			addLogEntry(logEntry);
		}
		
		return id;
	}
	
	
	void completedDataAccess(
		data_access_id_t &dataAccessId,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> CompletedDataAccess " << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void dataAccessBecomesRemovable(
		data_access_id_t &dataAccessId,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> DataAccessBecomesRemovable " << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void removedDataAccess(
		data_access_id_t &dataAccessId,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> RemoveDataAccess " << dataAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void linkedDataAccesses(
		data_access_id_t &sourceAccessId,
		task_id_t sinkTaskId, access_object_type_t sinkObjectType,
		DataAccessRegion region,
		bool direct,
		__attribute__((unused)) bool bidirectional,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> LinkDataAccesses " << sourceAccessId << " -> ";
		switch(sinkObjectType) {
			case regular_access_type:
				logEntry->_contents << " Access";
				break;
			case entry_fragment_type:
				logEntry->_contents << " Entry fragment";
				break;
			case taskwait_type:
				logEntry->_contents << " Taskwait";
				break;
			case top_level_sink_type:
				logEntry->_contents << " Top level sink";
				break;
		}
		logEntry->_contents << " from Task:" << sinkTaskId;
		
		logEntry->_contents << " [" << region << "]"
			<< (direct ? " direct" : "indirect")
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void unlinkedDataAccesses(
		data_access_id_t &sourceAccessId,
		task_id_t sinkTaskId, access_object_type_t sinkObjectType,
		bool direct,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> UnlinkDataAccesses " << sourceAccessId << " -> ";
		switch(sinkObjectType) {
			case regular_access_type:
				logEntry->_contents << " Access";
				break;
			case entry_fragment_type:
				logEntry->_contents << " Entry fragment";
				break;
			case taskwait_type:
				logEntry->_contents << " Taskwait";
				break;
			case top_level_sink_type:
				logEntry->_contents << " Top level sink";
				break;
		}
		logEntry->_contents << " from Task:" << sinkTaskId;
		
		logEntry->_contents << (direct ? " direct" : "indirect")
			<< " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void reparentedDataAccess(
		data_access_id_t &oldSuperAccessId,
		data_access_id_t &newSuperAccessId,
		data_access_id_t &dataAccessId,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> ReplaceSuperAccess " << dataAccessId << " " << oldSuperAccessId << "->" << newSuperAccessId << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void newDataAccessProperty(
		data_access_id_t &dataAccessId,
		char const *shortPropertyName,
		char const *longPropertyName,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> DataAccessNewProperty " << dataAccessId << " " << longPropertyName << " (" << shortPropertyName << ") triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
	
	void newDataAccessLocation(
		data_access_id_t &dataAccessId,
		MemoryPlace const *newLocation,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccessLinks) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> DataAccessNewLocation " << dataAccessId << " MemoryPlaceId:" << newLocation->getIndex() << " triggererTask:" << context._taskId;
		
		addLogEntry(logEntry);
	}
}
