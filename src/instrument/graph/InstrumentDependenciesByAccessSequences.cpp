#include <algorithm>
#include <cassert>

#include <dependencies/DataAccessSequence.hpp>

#include "InstrumentDependenciesByAccessSequences.hpp"

#include "InstrumentDataAccessId.hpp"
#include "InstrumentDataAccessSequenceId.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentGraph.hpp"
#include "executors/threads/WorkerThread.hpp"


namespace Instrument {
	using namespace Graph;
	
	data_access_sequence_id_t registerAccessSequence(data_access_id_t parentDataAccessId, task_id_t triggererTaskId)
	{
		if (!Graph::_showDependencyStructures) {
			return data_access_sequence_id_t(-1);
		}
		
		data_access_sequence_id_t dataAccessSequenceId = Graph::_nextDataAccessSequenceId++;
		
		if (parentDataAccessId != data_access_id_t()) {
			// See comment above call to getAccessSequence in next function
			access_sequence_t &accessSequence = getAccessSequence(dataAccessSequenceId, triggererTaskId);
			accessSequence._superAccess = parentDataAccessId;
		}
		
		return dataAccessSequenceId;
	}
	
	data_access_id_t addedDataAccessInSequence(
		data_access_sequence_id_t dataAccessSequenceId,
		DataAccessType accessType,
		bool satisfied,
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
		
		CPU *cpu = (CPU *) currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		data_access_id_t dataAccessId = Graph::_nextDataAccessId++;
		
		register_task_access_in_sequence_step_t *step = new register_task_access_in_sequence_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessSequenceId, dataAccessId, accessType, satisfied, originatorTaskId
		);
		_executionSequence.push_back(step);
		
		// Create the access sequence if necessary and the access, but in an almost uninitialized state.
		// It will be initialized when we replay the register_task_access_in_sequence_step_t created above
		// and will be modified as we replay the steps recorder in the remaining functions of this
		// instrumentation interface
		access_sequence_t &accessSequence = getAccessSequence(dataAccessSequenceId, originatorTaskId);
		
		// If the first access of a sequence, then record which is the corresponding access of the parent
		bool firstInSequence = accessSequence._accesses.empty();
		if (firstInSequence) {
			if (accessSequence._superAccess != data_access_id_t()) {
				_superAccessByAccess[dataAccessId] = accessSequence._superAccess;
				_firstSubAccessByAccess[accessSequence._superAccess] = dataAccessId;
			}
		}
		
		access_t &access = accessSequence._accesses[dataAccessId];
		access._originator = originatorTaskId;
		
		return dataAccessId;
	}
	
	
	void upgradedDataAccessInSequence(
		data_access_sequence_id_t dataAccessSequenceId,
		data_access_id_t dataAccessId,
		__attribute__((unused)) DataAccessType previousAccessType,
		DataAccessType newAccessType,
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
		
		CPU *cpu = (CPU *) currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		upgrade_task_access_in_sequence_step_t *step = new upgrade_task_access_in_sequence_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessSequenceId, dataAccessId, newAccessType, becomesUnsatisfied,
			originatorTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void dataAccessBecomesSatisfied(
		data_access_sequence_id_t dataAccessSequenceId,
		data_access_id_t dataAccessId,
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
		
		CPU *cpu = (CPU *) currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		task_access_in_sequence_becomes_satisfied_step_t *step = new task_access_in_sequence_becomes_satisfied_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessSequenceId, dataAccessId, triggererTaskId, targetTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void removedDataAccessFromSequence(
		data_access_sequence_id_t dataAccessSequenceId,
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
		
		CPU *cpu = (CPU *) currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		removed_task_access_from_sequence_step_t *step = new removed_task_access_from_sequence_step_t(
			cpu->_virtualCPUId, threadId,
			dataAccessSequenceId, dataAccessId, triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
}
