#include <atomic>
#include <cassert>

#include "InstrumentDependenciesByAccessLinks.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	data_access_id_t createdDataAccess(
		data_access_id_t superAccessId,
		DataAccessType accessType, bool weak, DataAccessRange range,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t originatorTaskId
	) {
		static std::atomic<data_access_id_t::inner_type_t> _nextDataAccessId(1);
		
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> CreateDataAccess " << id << " superaccess:" << superAccessId << " ";
		
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
		
		logEntry->_contents << " " << range;
		
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
		
		return id;
	}
	
	
	void upgradedDataAccess(
		data_access_id_t dataAccessId,
		DataAccessType previousAccessType,
		bool previousWeakness,
		DataAccessType newAccessType,
		bool newWeakness,
		bool becomesUnsatisfied,
		task_id_t triggererTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		
		logEntry->_contents << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void dataAccessBecomesSatisfied(
		data_access_id_t dataAccessId,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t triggererTaskId,
		task_id_t targetTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> DataAccessBecomesSatisfied " << dataAccessId << " triggererTask:" << triggererTaskId << " targetTask:" << targetTaskId;
		
		if (readSatisfied) {
			logEntry->_contents << " +read_safistied";
		}
		if (writeSatisfied) {
			logEntry->_contents << " +write_safistied";
		}
		if (globallySatisfied) {
			logEntry->_contents << " +safistied";
		}
		if (!readSatisfied && !writeSatisfied && !globallySatisfied) {
			logEntry->_contents << " remains_unsatisfied";
		}
		
		addLogEntry(logEntry);
	}
	
	
	void removedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> RemoveDataAccessFromSequence " << dataAccessId << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void linkedDataAccesses(
		data_access_id_t sourceAccessId,
		data_access_id_t sinkAccessId,
		DataAccessRange range,
		bool direct,
		__attribute__((unused)) bool bidirectional,
		task_id_t triggererTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> LinkDataAccesses " << sourceAccessId << " -> " << sinkAccessId << " [" << range << "]" << (direct ? " direct" : "indirect") << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void unlinkedDataAccesses(
		data_access_id_t sourceAccessId,
		data_access_id_t sinkAccessId,
		bool direct,
		task_id_t triggererTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> UnlinkDataAccesses " << sourceAccessId << " -> " << sinkAccessId << (direct ? " direct" : "indirect") << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
	void reparentedDataAccess(
		data_access_id_t oldSuperAccessId,
		data_access_id_t newSuperAccessId,
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		if (!_verboseDependenciesByAccessLinks) {
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
		logEntry->_contents << " <-> ReplaceSuperAccess " << dataAccessId << " " << oldSuperAccessId << "->" << newSuperAccessId << " triggererTask:" << triggererTaskId;
		
		addLogEntry(logEntry);
	}
	
	
}
