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
				bool becomesSatisfied = dataAccessSequence->reevaluateSatisfactibility(nextPosition);
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

