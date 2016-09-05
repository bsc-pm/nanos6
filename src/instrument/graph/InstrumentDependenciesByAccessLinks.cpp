#include <algorithm>
#include <cassert>

#include "InstrumentDependenciesByAccessLinks.hpp"

#include "InstrumentDataAccessId.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentGraph.hpp"
#include "executors/threads/WorkerThread.hpp"


namespace Instrument {
	using namespace Graph;
	
	data_access_id_t createdDataAccess(
		data_access_id_t superAccessId,
		DataAccessType accessType, bool weak, DataAccessRange range,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t originatorTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return data_access_id_t(-1);
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		data_access_id_t dataAccessId = Graph::_nextDataAccessId++;
		
		create_data_access_step_t *step = new create_data_access_step_t(
			cpu->_virtualCPUId, threadId,
			superAccessId, dataAccessId, accessType, range, weak,
			readSatisfied, writeSatisfied, globallySatisfied,
			originatorTaskId
		);
		_executionSequence.push_back(step);
		
		// Create the access but in an almost uninitialized state. It will be initialized when we
		// replay the create_data_access_step_t created above and will be modified as we replay the
		// steps recorded in the remaining functions of this instrumentation interface
		access_t &access = getAccess(dataAccessId);
		access._superAccess = superAccessId;
		access._originator = originatorTaskId;
		
		task_info_t &taskInfo = _taskToInfoMap[originatorTaskId];
		
		// The "main" function is not supposed to have dependencies
		assert(taskInfo._parent != -1);
		
		task_info_t &parentInfo = _taskToInfoMap[taskInfo._parent];
		
		assert(!parentInfo._phaseList.empty());
		auto it = parentInfo._phaseList.end();
		it--;
		
		phase_t *lastPhase = *it;
		task_group_t *lastTaskGroup = dynamic_cast<task_group_t *> (lastPhase);
		
		assert(lastTaskGroup != nullptr);
		lastTaskGroup->_dataAccesses.insert(dataAccessId);
		
		return dataAccessId;
	}
	
	
	void upgradedDataAccess(
		data_access_id_t dataAccessId,
		__attribute__((unused)) DataAccessType previousAccessType,
		__attribute__((unused)) bool previousWeakness,
		DataAccessType newAccessType,
		bool newWeakness,
		bool becomesUnsatisfied,
		task_id_t originatorTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		upgrade_data_access_step_t *step = new upgrade_data_access_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessId,
			newAccessType, newWeakness,
			becomesUnsatisfied,
			originatorTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void dataAccessBecomesSatisfied(
		data_access_id_t dataAccessId,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t triggererTaskId,
		task_id_t targetTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		data_access_becomes_satisfied_step_t *step = new data_access_becomes_satisfied_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessId,
			readSatisfied, writeSatisfied, globallySatisfied,
			triggererTaskId, targetTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void removedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		removed_data_access_step_t *step = new removed_data_access_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessId, triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void linkedDataAccesses(
		data_access_id_t sourceAccessId, data_access_id_t sinkAccessId,
		DataAccessRange range,
		bool direct, bool bidirectional,
		task_id_t triggererTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		access_t &sourceAccess = getAccess(sourceAccessId);
		access_t &sinkAccess = getAccess(sinkAccessId);
		sourceAccess._nextLinks.emplace(
			std::pair<data_access_id_t, link_to_next_t> (sinkAccessId, link_to_next_t(direct, bidirectional))
		); // A "not created" link
		sinkAccess._previousLinks.insert(sourceAccessId);
		
		linked_data_accesses_step_t *step = new linked_data_accesses_step_t(
			cpu->_virtualCPUId, threadId,
			sourceAccessId, sinkAccessId,
			range,
			direct, bidirectional,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void unlinkedDataAccesses(
		data_access_id_t sourceAccessId,
		data_access_id_t sinkAccessId,
		bool direct,
		task_id_t triggererTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		unlinked_data_accesses_step_t *step = new unlinked_data_accesses_step_t(
			cpu->_virtualCPUId, threadId,
			sourceAccessId, sinkAccessId, direct,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void reparentedDataAccess(
		data_access_id_t oldSuperAccessId,
		data_access_id_t newSuperAccessId,
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		if (!Graph::_showDependencyStructures) {
			return;
		}
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		CPU *cpu = (CPU *) currentThread->getComputePlace();
		assert(cpu != nullptr);
		
		reparented_data_access_step_t *step = new reparented_data_access_step_t(
			cpu->_virtualCPUId, threadId,
			oldSuperAccessId, newSuperAccessId, dataAccessId,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
}
