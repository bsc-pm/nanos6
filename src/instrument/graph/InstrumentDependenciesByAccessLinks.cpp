#include <algorithm>
#include <cassert>

#include "ExecutionSteps.hpp"
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
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		data_access_id_t dataAccessId = Graph::_nextDataAccessId++;
		
		create_data_access_step_t *step = new create_data_access_step_t(
			cpuId, threadId,
			superAccessId, dataAccessId, accessType, range, weak,
			readSatisfied, writeSatisfied, globallySatisfied,
			originatorTaskId
		);
		_executionSequence.push_back(step);
		
		task_info_t &taskInfo = _taskToInfoMap[originatorTaskId];
		
		access_t *access = new access_t();
		access->_id = dataAccessId;
		access->_superAccess = superAccessId;
		access->_originator = originatorTaskId;
		access->_firstGroupAccess = dataAccessId;
		
		// We need the final range and type of each access to calculate the full graph
		access->_type = accessType;
		access->_accessRange = range;
		
		_accessIdToAccessMap[dataAccessId] = access;
		
		taskInfo._allAccesses.insert(access);
		taskInfo._liveAccesses.insert(AccessWrapper(access));
		
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
		if (dataAccessId == data_access_id_t()) {
			// A data access that has not been fully created yet
			return;
		}
		
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		upgrade_data_access_step_t *step = new upgrade_data_access_step_t(
			cpuId, threadId,
			dataAccessId,
			newAccessType, newWeakness,
			becomesUnsatisfied,
			originatorTaskId
		);
		_executionSequence.push_back(step);
		
		// We need the final type of each access to calculate the full graph
		access_t *access = _accessIdToAccessMap[dataAccessId];
		assert(access != nullptr);
		access->_type = newAccessType;
	}
	
	
	void dataAccessBecomesSatisfied(
		data_access_id_t dataAccessId,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t triggererTaskId,
		task_id_t targetTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		data_access_becomes_satisfied_step_t *step = new data_access_becomes_satisfied_step_t(
			cpuId, threadId,
			dataAccessId,
			readSatisfied, writeSatisfied, globallySatisfied,
			triggererTaskId, targetTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void modifiedDataAccessRange(
		data_access_id_t dataAccessId,
		DataAccessRange newRange,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		modified_data_access_range_step_t *step = new modified_data_access_range_step_t(
			cpuId, threadId,
			dataAccessId,
			newRange,
			triggererTaskId
		);
		_executionSequence.push_back(step);
		
		// We need the final range of each access to calculate the full graph
		access_t *access = _accessIdToAccessMap[dataAccessId];
		assert(access != nullptr);
		access->_accessRange = newRange;
	}
	
	
	data_access_id_t fragmentedDataAccess(
		data_access_id_t dataAccessId,
		DataAccessRange newRange,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		access_t *originalAccess = _accessIdToAccessMap[dataAccessId];
		assert(originalAccess != nullptr);
		
		data_access_id_t newDataAccessId = Graph::_nextDataAccessId++;
		
		fragment_data_access_step_t *step = new fragment_data_access_step_t(
			cpuId, threadId,
			dataAccessId, newDataAccessId, newRange,
			triggererTaskId
		);
		_executionSequence.push_back(step);
		
		if (!originalAccess->fragment()) {
			task_info_t &taskInfo = _taskToInfoMap[originalAccess->_originator];
			
			// Copy all the contents so that we also get any already existing link
			access_t *newAccess = new access_t();
			*newAccess = *originalAccess;
			newAccess->_accessRange = newRange;
			
			newAccess->_id = newDataAccessId;
			
			taskInfo._allAccesses.insert(newAccess);
			taskInfo._liveAccesses.insert(AccessWrapper(newAccess));
			
			_accessIdToAccessMap[newDataAccessId] = newAccess;
		} else {
			access_fragment_t *originalFragment = (access_fragment_t *) originalAccess;
			
			task_group_t *taskGroup = originalFragment->_taskGroup;
			assert(taskGroup != nullptr);
			
			// Copy all the contents so that we also get any already existing link
			access_fragment_t *newFragment = new access_fragment_t();
			*newFragment = *originalFragment;
			newFragment->_accessRange = newRange;
			
			newFragment->_id = newDataAccessId;
			
			// Fragments are inserted in the task group that corresponds to the phase in which they are created
			taskGroup->_allFragments.insert(newFragment);
			taskGroup->_liveFragments.insert(AccessFragmentWrapper(newFragment));
			
			_accessIdToAccessMap[newDataAccessId] = newFragment;
		}
		
		// Link the new access/fragment into the access group
		originalAccess->_nextGroupAccess = newDataAccessId;
		
		return newDataAccessId;
	}
	
	
	data_access_id_t createdDataSubaccessFragment(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		access_t *originalAccess = _accessIdToAccessMap[dataAccessId];
		assert(originalAccess != nullptr);
		
		data_access_id_t newDataAccessId = Graph::_nextDataAccessId++;
		
		create_subaccess_fragment_step_t *step = new create_subaccess_fragment_step_t(
			cpuId, threadId,
			dataAccessId, newDataAccessId,
			triggererTaskId
		);
		_executionSequence.push_back(step);
		
		task_info_t &taskInfo = _taskToInfoMap[originalAccess->_originator];
		
		// The last phase of the creator task should be a taskgroup that includes the new task that
		// triggers the creation of the subaccess fragment
		task_group_t *taskGroup = nullptr;
		if (!taskInfo._phaseList.empty()) {
			phase_t *lastPhase = taskInfo._phaseList.back();
			taskGroup = dynamic_cast<task_group_t *>(lastPhase);
		}
		assert(taskGroup != nullptr);
		
		// Create the fragment
		access_fragment_t *fragment = new access_fragment_t();
		fragment->_id = newDataAccessId;
		fragment->_superAccess = originalAccess->_superAccess;
		fragment->_originator = originalAccess->_originator;
		fragment->fragment() = true;
		fragment->_firstGroupAccess = newDataAccessId;
		fragment->_nextGroupAccess = data_access_id_t();
		fragment->_taskGroup = taskGroup;
		
		taskGroup->_allFragments.insert(fragment);
		taskGroup->_liveFragments.insert(AccessFragmentWrapper(fragment));
		
		_accessIdToAccessMap[newDataAccessId] = fragment;
		
		return newDataAccessId;
	}
	
	
	void completedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		completed_data_access_step_t *step = new completed_data_access_step_t(
			cpuId, threadId,
			dataAccessId,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void dataAccessBecomesRemovable(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		data_access_becomes_removable_step_t *step = new data_access_becomes_removable_step_t(
			cpuId, threadId,
			dataAccessId, triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void removedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		removed_data_access_step_t *step = new removed_data_access_step_t(
			cpuId, threadId,
			dataAccessId, triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void linkedDataAccesses(
		data_access_id_t sourceAccessId, task_id_t sinkTaskId,
		DataAccessRange range,
		bool direct, bool bidirectional,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		access_t *sourceAccess = _accessIdToAccessMap[sourceAccessId];
		assert(sourceAccess != nullptr);
		sourceAccess->_nextLinks.emplace(
			std::pair<task_id_t, link_to_next_t> (sinkTaskId, link_to_next_t(direct, bidirectional))
		); // A "not created" link
		
		linked_data_accesses_step_t *step = new linked_data_accesses_step_t(
			cpuId, threadId,
			sourceAccessId, sinkTaskId,
			range,
			direct, bidirectional,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void unlinkedDataAccesses(
		data_access_id_t sourceAccessId,
		task_id_t sinkTaskId,
		bool direct,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		unlinked_data_accesses_step_t *step = new unlinked_data_accesses_step_t(
			cpuId, threadId,
			sourceAccessId, sinkTaskId, direct,
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
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		reparented_data_access_step_t *step = new reparented_data_access_step_t(
			cpuId, threadId,
			oldSuperAccessId, newSuperAccessId, dataAccessId,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}
	
	
	void newDataAccessProperty(
		data_access_id_t dataAccessId,
		char const *shortPropertyName,
		char const *longPropertyName,
		task_id_t triggererTaskId
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		new_data_access_property_step_t *step = new new_data_access_property_step_t(
			cpuId, threadId,
			dataAccessId,
			shortPropertyName, longPropertyName,
			triggererTaskId
		);
		_executionSequence.push_back(step);
	}

}
