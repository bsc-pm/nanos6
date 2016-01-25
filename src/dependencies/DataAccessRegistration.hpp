#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP


#include <cassert>
#include <deque>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessSequences.hpp>
#include <InstrumentTaskId.hpp>


class DataAccessRegistration {
private:
	typedef std::deque<Task *> satisfied_originator_list_t;
	
	static inline void unregisterDataAccess(Instrument::task_id_t instrumentationTaskId, DataAccess *dataAccess, satisfied_originator_list_t /* OUT */ &satisfiedOriginators)
	{
		DataAccessSequence *dataAccessSequence = dataAccess->_dataAccessSequence;
		DataAccessSequence::access_sequence_t::iterator dataAccessPosition = dataAccessSequence->_accessSequence.iterator_to(*dataAccess);
		
		DataAccess *superAccess = 0;
		bool superAccessCompleted = false;
		
		// Erase the DataAccess and reevaluate if the following ones in the sequence become satisfied
		// NOTE: This is done with the lock held
		{
			std::lock_guard<SpinLock> guard(dataAccessSequence->_lock);
			
			Instrument::removedDataAccessFromSequence(
				dataAccessSequence->_instrumentationId,
				dataAccess->_instrumentationId,
				instrumentationTaskId
			);
			auto nextPosition = dataAccessSequence->_accessSequence.erase(dataAccessPosition);
			
			while (nextPosition != dataAccessSequence->_accessSequence.end()) {
				bool becomesSatisfied = dataAccessSequence->reevaluateSatisfiability(nextPosition);
				if (becomesSatisfied) {
					Instrument::dataAccessBecomesSatisfied(
						dataAccessSequence->_instrumentationId,
						nextPosition->_instrumentationId,
						instrumentationTaskId,
						nextPosition->_originator->getInstrumentationTaskId()
					);
					satisfiedOriginators.push_back(nextPosition->_originator);
					
					nextPosition++;
				} else {
					// Either it was already satisfied or it cannot become satisfied
					break;
				}
			}
			
			if (dataAccessSequence->_accessSequence.empty()) {
				superAccess = dataAccessSequence->_superAccess;
				if (superAccess != 0) {
					superAccessCompleted = (--superAccess->_completionCountdown == 0);
				}
			}
		}
		
		if (superAccessCompleted) {
			assert(superAccess != 0);
			Task *superOriginator = superAccess->_originator;
			assert(superOriginator != 0);
			
			unregisterDataAccess(superOriginator->getInstrumentationTaskId(), superAccess, satisfiedOriginators);
		}
	}
	
	
	//! Process all the originators for whose a DataAccess has become satisfied
	static inline void processSatisfiedOriginators(satisfied_originator_list_t &satisfiedOriginators, HardwarePlace *hardwarePlace)
	{
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			bool becomesReady = satisfiedOriginator->decreasePredecessors();
			if (becomesReady) {
				HardwarePlace *idleHardwarePlace = Scheduler::addReadyTask(satisfiedOriginator, hardwarePlace);
				
				if (idleHardwarePlace != nullptr) {
					ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
				}
			}
		}
	}
	
	
public:

	//! \brief adds a task access to the end of a sequence taking into account repeated accesses
	//! 
	//! \param[in] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] accessSequence the sequnece on which to add the new access
	//! \param[out] dataAccess gets initialized with a pointer to the new DataAccess object or null if there was already a previous one for that task
	//! 
	//! \returns true is the access can be started
	//!
	//! The new DataAccess object has the task as its originator and is inserted in the DataAccessSequence.
	//! However, it is not inserted in the list of accesses of the Task.
	//! 
	//! If the task has already a previous access, it may be upgraded if necessary, and dataAccess is set to null. The return
	//! value indicates if the new access produces an additional dependency (only possible if the previous one did not).
	static inline bool registerTaskDataAccess(Task *task, DataAccessType accessType, DataAccessSequence *accessSequence, DataAccess *&dataAccess)
	{
		assert(task != 0);
		assert(accessSequence != nullptr);
		std::lock_guard<SpinLock> guard(accessSequence->_lock);
		
		auto position = accessSequence->_accessSequence.rbegin();
		
		// If there are no previous accesses, then the new access can be satisfied
		bool satisfied = (position == accessSequence->_accessSequence.rend());
		
		if (position != accessSequence->_accessSequence.rend()) {
			// There is a "last" access
			DataAccess &lastAccess = *position;
			
			if (lastAccess._originator == task) {
				// The task "accesses" twice to the same location
				
				dataAccess = 0;
				return accessSequence->upgradeAccess(task, position, lastAccess, accessType);
			} else {
				if ((lastAccess._type == WRITE_ACCESS_TYPE) || (lastAccess._type == READWRITE_ACCESS_TYPE)) {
					satisfied = false;
				} else {
					satisfied = (lastAccess._type == accessType) && lastAccess._satisfied;
				}
			}
		} else {
			// We no longer have (or never had) information about any previous access to this storage
			Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), accessSequence, true);
		}
		
		if (accessSequence->_accessSequence.empty()) {
			accessSequence->_instrumentationId = Instrument::registerAccessSequence((accessSequence->_superAccess != 0 ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()), task->getInstrumentationTaskId());
			if (accessSequence->_superAccess != 0) {
				// The access of the parent will start having subaccesses
				
				// 1. The parent is adding this task, so it cannot have finished (>=1)
				// 2. The sequence is empty, so it has not been counted yet (<2)
				assert(accessSequence->_superAccess->_completionCountdown.load() == 1);
				
				accessSequence->_superAccess->_completionCountdown++;
			}
		}
		
		Instrument::data_access_id_t dataAccessInstrumentationId =
		Instrument::addedDataAccessInSequence(accessSequence->_instrumentationId, accessType, satisfied, task->getInstrumentationTaskId());
		Instrument::addTaskToAccessGroup(accessSequence, task->getInstrumentationTaskId());
		
		dataAccess = new DataAccess(accessSequence, accessType, satisfied, task, accessSequence->_accessRange, dataAccessInstrumentationId);
		accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
		
		return satisfied;
	}
	
	
	//! \brief Performs the task dependency registration procedure
	//! 
	//! \param[in] task the Task whose dependencies need to be calculated
	//! 
	//! \returns true if the task is already ready
	static inline bool registerTaskDataAccesses(Task *task)
	{
		assert(task != 0);
		
		// We increase the number of predecessors to avoid having the task become ready while we are adding its dependencies.
		// We do it by 2 because we add the data access and unlock access to it before increasing the number of predecessors.
		task->increasePredecessors(2);
		
		nanos_task_info *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		taskInfo->register_depinfo(task, task->getArgsBlock());
		
		return task->decreasePredecessors(2);
	}
	
	
	static inline void unregisterTaskDataAccesses(Task *finishedTask)
	{
		assert(finishedTask != 0);
		
		assert(WorkerThread::getCurrentWorkerThread() != 0);
		HardwarePlace *hardwarePlace = WorkerThread::getCurrentWorkerThread()->getHardwarePlace();
		assert(hardwarePlace != 0);
		
		DataAccess *dataAccess = finishedTask->popDataAccess();
		
		// A temporary list of tasks to minimize the time spent with the mutex held.
		satisfied_originator_list_t satisfiedOriginators; // NOTE: This could be moved as a member of the WorkerThread for efficiency.
		while (dataAccess != 0) {
			assert(dataAccess->_originator == finishedTask);
			bool canRemoveAccess = (--dataAccess->_completionCountdown == 0);
			if (canRemoveAccess) {
				unregisterDataAccess(finishedTask->getInstrumentationTaskId(), dataAccess, /* OUT */ satisfiedOriginators);
				processSatisfiedOriginators(satisfiedOriginators, hardwarePlace);
			}
			
			dataAccess = finishedTask->popDataAccess();
			satisfiedOriginators.clear();
		}
	}
};


#endif // DATA_ACCESS_REGISTRATION_HPP

