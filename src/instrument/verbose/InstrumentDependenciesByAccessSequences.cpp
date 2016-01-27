#include <atomic>
#include <cassert>

#include "InstrumentDependenciesByAccessSequences.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	data_access_sequence_id_t registerAccessSequence(data_access_id_t parentDataAccessId, task_id_t triggererTaskId) {
		static std::atomic<data_access_sequence_id_t::inner_type_t> _nextDataAccessSequenceId(1);
		
		if (!_verboseDependenciesByAccessSequence) {
			return data_access_sequence_id_t();
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		data_access_sequence_id_t id = _nextDataAccessSequenceId++;
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> RegisterAccessSequence " << id << " parent:" << parentDataAccessId << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
		
		return id;
	}
	
	
	data_access_id_t addedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, DataAccessType accessType, bool weak, bool satisfied, task_id_t originatorTaskId) {
		static std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId(1);
		
		if (!_verboseDependenciesByAccessSequence) {
			return data_access_id_t();
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		data_access_id_t id = _nextDataAccessId++;
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> AddDataAccessToSequence " << id << " sequence:" << dataAccessSequenceId << " ";
		
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
			default:
				logEntry->_contents << " unknown_access_type";
				break;
		}
		
		if (satisfied) {
			logEntry->_contents << " safistied";
		} else {
			logEntry->_contents << " unsatisfied";
		}
		
		logEntry->_contents << " originator:" << originatorTaskId;
		
		addLogEntry(logEntry);
		
		return id;
	}
	
	
	void upgradedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, DataAccessType previousAccessType, bool previousWeakness, DataAccessType newAccessType, bool newWeakness, bool becomesUnsatisfied, task_id_t triggererTaskId) {
		if (!_verboseDependenciesByAccessSequence) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> UpgradeDataAccess " << dataAccessId << " sequence:" << dataAccessSequenceId;
		
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
		
		logEntry->_contents << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}


	void dataAccessBecomesSatisfied(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId, task_id_t targetTaskId) {
		if (!_verboseDependenciesByAccessSequence) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> DataAccessBecomesSatisfied " << dataAccessId << " sequence:" << dataAccessSequenceId << " triggererTask:" << triggererTaskId << " targetTask:" << targetTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void removedDataAccessFromSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId) {
		if (!_verboseDependenciesByAccessSequence) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> RemoveDataAccessFromSequence " << dataAccessId << " sequence:" << dataAccessSequenceId << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void replacedSequenceOfDataAccess(data_access_sequence_id_t previousDataAccessSequenceId, data_access_sequence_id_t newDataAccessSequenceId, data_access_id_t dataAccessId, data_access_id_t beforeDataAccessId, task_id_t triggererTaskId) {
		if (!_verboseDependenciesByAccessSequence) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-> ReplaceSequenceOfDataAccess " << dataAccessId << " " << previousDataAccessSequenceId << "->" << newDataAccessSequenceId << " beforeAccess:" << beforeDataAccessId << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
}
